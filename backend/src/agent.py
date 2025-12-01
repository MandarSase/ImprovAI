#!/usr/bin/env python3
import asyncio
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
from dotenv import load_dotenv
import random

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    RoomInputOptions,
    cli,
    RunContext,
    function_tool,
)

from livekit.plugins import deepgram, google, murf, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Config
# -------------------------
SCENARIO_TIMEOUT = 6        # seconds to wait for scenario generation
REACTION_TIMEOUT = 5        # seconds to wait for reaction generation
FINAL_SUMMARY_TIMEOUT = 8   # seconds to wait for final summary generation
TRIM_CHARS = 280            # hard limit chars for any reply from agent
END_SCENE_PAUSE = 8         # pause seconds after user says end scene

# -------------------------
# Logging
# -------------------------
logger = logging.getLogger("improv_ai")
logger.setLevel(logging.INFO)
h = logging.StreamHandler()
h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(h)

load_dotenv(".env.local")

# -------------------------
# Paths
# -------------------------
BASE = Path(__file__).resolve().parent
SHARED = BASE / "shared-data"
SESSION_DIR = SHARED / "improv_sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)
SCENARIO_FILE = SHARED / "scenarios.json"

# -------------------------
# Dataclass for state
# -------------------------
@dataclass
class ImprovState:
    player_name: Optional[str] = None
    current_round: int = 0
    max_rounds: int = 3
    rounds: List[Dict[str, str]] = None  # Each: {"scenario":..., "player":..., "reaction":...}
    phase: str = "ask_name"  # ask_name | intro | scenario | await_improv | reacting | ask_continue | done

    def __post_init__(self):
        if self.rounds is None:
            self.rounds = []

# Helper container
class UD:
    pass

# -------------------------
# Helpers
# -------------------------
def session_file_for(name: str) -> Path:
    safe = name.lower().strip().replace(" ", "_")
    return SESSION_DIR / f"{safe}.json"

def trim_reply(text: str, max_chars: int = TRIM_CHARS) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # try to cut on a sentence or word boundary
    truncated = text[:max_chars]
    # cut back to last punctuation or space
    m = re.search(r"[.!?]\s+[A-Z][^.!?]*$", truncated)
    if m:
        return truncated[:m.start(0)].strip() + "..."
    # else cut to last space
    last_space = truncated.rfind(" ")
    if last_space > int(max_chars * 0.6):
        return truncated[:last_space].rstrip() + "..."
    return truncated.rstrip() + "..."

async def llm_generate_with_timeout(llm, prompt: str, timeout: int):
    """
    Calls the llm.generate with an asyncio timeout and returns a plain string.
    Accepts both string returns and objects with .text.
    """
    try:
        coro = llm.generate(prompt=prompt)
        raw = await asyncio.wait_for(coro, timeout=timeout)
        return raw.text.strip() if hasattr(raw, "text") else str(raw).strip()
    except asyncio.TimeoutError:
        logger.warning("LLM generation timed out")
        raise
    except Exception as e:
        logger.warning(f"LLM generation failed: {e}")
        raise

def detect_end_scene(text: str) -> bool:
    """
    Robust detection of 'end scene' variants. Matches:
    - 'end scene'
    - 'endscene'
    - 'end scene.' with quotes or punctuation
    - handles noisy inputs by searching for the token pair
    """
    if not text:
        return False
    normalized = text.lower()
    # common patterns
    if re.search(r"\bend\s*scene\b", normalized):
        return True
    # tolerate some noise like endscene or end-scene
    if re.search(r"\bend[-_]?\s*scene\b", normalized):
        return True
    return False

# -------------------------
# Tools
# -------------------------
@function_tool
async def set_player_name(ctx: RunContext, name: str) -> str:
    ud: UD = ctx.userdata
    name = name.strip()
    ud.state.player_name = name
    # ensure summary field on state
    if not hasattr(ud.state, "summary"):
        ud.state.summary = ""

    file = session_file_for(name)
    if file.exists():
        ud.state.phase = "intro"
        return f"Welcome back, {name}! Would you like to continue your previous Improv Battle or start fresh?"
    else:
        ud.state.phase = "intro"
        return f"Nice to meet you, {name}! Are you ready to jump into Improv Battle and start your first scenario?"

@function_tool
async def continue_or_new(ctx: RunContext, choice: str) -> str:
    ud: UD = ctx.userdata
    choice = (choice or "").strip().lower()
    file = session_file_for(ud.state.player_name)

    if choice in ("continue", "resume"):
        if file.exists():
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            ud.state = ImprovState(
                player_name=data.get("player_name"),
                current_round=data.get("current_round", 0),
                max_rounds=data.get("max_rounds", 3),
                rounds=data.get("rounds", []),
                phase="scenario"
            )
            ud.state.summary = data.get("summary", "")
            return f"Resuming your improv battle, {ud.state.player_name}! Let's jump back in!"
        else:
            ud.state.phase = "scenario"
            return "Hmm, I couldn't find any saved progress. Let's start fresh! Ready?"
    else:
        ud.state = ImprovState(player_name=ud.state.player_name)
        ud.state.summary = ""
        ud.state.phase = "scenario"
        return "Starting a brand-new Improv Battle! Let's go!"

@function_tool
async def generate_scenario(ctx: RunContext) -> str:
    ud: UD = ctx.userdata
    ud.state.phase = "await_improv"

    logger.info(f"Generating scenario for {ud.state.player_name} (Round {ud.state.current_round + 1})")

    prompt = f"""
You are the witty host of an improv TV show called "Improv Battle".
Generate a short improv scenario for the player.
Requirements:
- Assign the player a role.
- Describe the situation.
- Add tension, absurdity, or humor.
- Keep it 1–2 sentences max.

Player name: {ud.state.player_name}
"""

    scenario_text = None
    try:
        scenario_text = await llm_generate_with_timeout(ud.session.llm, prompt, SCENARIO_TIMEOUT)
        if not scenario_text:
            raise ValueError("Empty scenario from LLM")
        scenario_text = trim_reply(scenario_text)
        logger.info(f"LLM scenario: {scenario_text}")
    except Exception as e:
        logger.warning(f"Falling back to local scenarios because: {e}")
        try:
            with open(SCENARIO_FILE, "r", encoding="utf-8") as f:
                scenarios = json.load(f)
            scenario_text = random.choice(scenarios)
        except Exception as e2:
            logger.error(f"Failed to load local scenarios.json: {e2}")
            scenario_text = "You are a brave explorer facing a mysterious puzzle. Figure it out quickly!"

    ud.state.rounds.append({"scenario": scenario_text, "player": "", "reaction": ""})
    # increment round number (we consider round starts when scenario is issued)
    ud.state.current_round += 1

    reply = f"Round {ud.state.current_round}! Your scenario is:\n\n{scenario_text}\n\nAct it out! When you're done, say 'End scene'."
    return trim_reply(reply, max_chars=TRIM_CHARS)

@function_tool
async def record_player_improv(ctx: RunContext, improv_text: str) -> str:
    ud: UD = ctx.userdata
    improv_text = (improv_text or "").strip()

    # Safety: ensure there's at least one round
    if not ud.state.rounds:
        return "I haven't given you a scenario yet. Say 'start' to get your first scenario."

    # If the user indicates "end scene" within their utterance, handle it
    if detect_end_scene(improv_text):
        # Clean the user's input by removing the end-scene token for storing the player text
        cleaned = re.sub(r'(?i)\bend[-_]?\s*scene\b', '', improv_text).strip()
        # If after removing the token there's meaningful text, keep it; else placeholder
        ud.state.rounds[-1]["player"] = cleaned if cleaned else "[no improv text provided]"
        ud.state.phase = "reacting"

        # dramatic pause before reaction
        await asyncio.sleep(END_SCENE_PAUSE)
        return await generate_reaction(ctx)

    # Normal recording: just store the text and wait for explicit 'end scene' or a user-triggered generation
    ud.state.rounds[-1]["player"] = improv_text
    ud.state.phase = "await_improv"  # player performing
    return "Got it — when you're finished say 'End scene' and I'll react."

@function_tool
async def generate_reaction(ctx: RunContext) -> str:
    ud: UD = ctx.userdata

    if not ud.state.rounds:
        return "There's nothing to react to right now."

    last = ud.state.rounds[-1]
    scenario = last.get("scenario", "")
    player_text = last.get("player", "")

    prompt = f"""
You are the witty host of a TV improv show.
Give a short reaction to this player's improv performance.

Scenario: {scenario}
Player's performance: {player_text}

Your reaction should be:
- Supportive, neutral, or mildly critical (random).
- Funny, energetic, TV-host style.
- 1–2 short sentences (keep it concise).
"""

    reaction_text = ""
    try:
        reaction_text = await llm_generate_with_timeout(ud.session.llm, prompt, REACTION_TIMEOUT)
        reaction_text = trim_reply(reaction_text)
    except Exception as e:
        logger.warning(f"Reaction LLM failed: {e}. Using fallback short reactions.")
        # fallback short reactions
        fallbacks = [
            "What a wild turn — loved the commitment!",
            "Nice character work — that sold the scene.",
            "Hilarious! Your timing was perfect.",
            "You leaned into the absurd and owned it."
        ]
        reaction_text = random.choice(fallbacks)

    # store reaction
    ud.state.rounds[-1]["reaction"] = reaction_text

    # accumulate concatenated summary
    if not hasattr(ud.state, "summary") or ud.state.summary is None:
        ud.state.summary = ""
    ud.state.summary += f"Round {ud.state.current_round}: {reaction_text}\n\n"

    # save round-wise immediately
    await save_progress(ctx)

    # After reaction, check whether we're at final round
    if ud.state.current_round >= ud.state.max_rounds:
        # produce final summary (with timeout)
        final_text = ""
        try:
            final_text = await llm_generate_with_timeout(
                ud.session.llm,
                f"You are the host of Improv Battle. Write a concise final summary of this player's performance across rounds.\n\nRounds data:\n{json.dumps(ud.state.rounds, indent=2)}",
                FINAL_SUMMARY_TIMEOUT
            )
            final_text = trim_reply(final_text, max_chars=TRIM_CHARS * 2)
        except Exception as e:
            logger.warning(f"Final summary LLM failed: {e}")
            final_text = "You were fantastic — full of creativity and bold choices. Great job!"

        # append final summary to accumulated summary
        ud.state.summary += "\n=== Final Summary ===\n" + final_text

        # Save final summary into the player's file (overwrite safely)
        file = session_file_for(ud.state.player_name)
        try:
            with open(file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["summary"] = ud.state.summary
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
        except Exception as e:
            # fallback: write fresh file
            logger.error(f"Failed to append final summary into file: {e}. Writing fresh file.")
            await save_progress(ctx)  # ensures rounds are saved
            file = session_file_for(ud.state.player_name)
            with open(file, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data["summary"] = ud.state.summary
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()

        ud.state.phase = "done"
        return trim_reply(reaction_text + "\n\n🎉 That's the end of the improv battle!\n\n" + final_text)

    # not final round: prompt user to continue
    ud.state.phase = "ask_continue"
    return trim_reply(reaction_text + "\n\nDo you want to continue to the next round, or save & exit?")

@function_tool
async def save_progress(ctx: RunContext) -> str:
    ud: UD = ctx.userdata
    file = session_file_for(ud.state.player_name)

    data = {
        "player_name": ud.state.player_name,
        "current_round": ud.state.current_round,
        "max_rounds": ud.state.max_rounds,
        "rounds": ud.state.rounds,
        "summary": getattr(ud.state, "summary", "")
    }

    # write atomically
    tmp = file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(file)

    logger.info(f"Saved progress for {ud.state.player_name} (round {ud.state.current_round})")
    return f"Progress saved for {ud.state.player_name} after round {ud.state.current_round}."

@function_tool
async def final_summary(ctx: RunContext) -> str:
    ud: UD = ctx.userdata
    ud.state.phase = "done"

    prompt = f"""
You are the host of Improv Battle.
Write a final summary of the player's performance across all rounds.

Include:
- Their improv style (character-driven, absurd, emotional, etc.)
- Specific standout moments from different rounds.
- A fun, energetic closing line.

Rounds data:
{json.dumps(ud.state.rounds, indent=2)}
"""

    try:
        summary_text = await llm_generate_with_timeout(ud.session.llm, prompt, FINAL_SUMMARY_TIMEOUT)
        return trim_reply(summary_text, max_chars=TRIM_CHARS * 2)
    except Exception as e:
        logger.error(f"Final summary generation failed: {e}")
        return "Couldn't generate final summary. But you rocked the improv battle!"

# -------------------------
# Agent Behavior
# -------------------------
class ImprovAgent(Agent):
    def __init__(self):
        instructions = """
You are the high-energy, witty host of a TV improv game show called "Improv Battle."
You run up to max_rounds of improv. After each round you:
- React to the performance (brief).
- Save the round and append a short summary line to the player's summary field.
- If final round reached, generate and append a final summary.

Trigger rules (guide only):
- When user gives name → use set_player_name
- When user says continue/start over → use continue_or_new
- When phase=await_improv and user performs → use record_player_improv
- When user says “end scene” → use record_player_improv (handled inside)
"""
        super().__init__(instructions=instructions, tools=[
            set_player_name,
            continue_or_new,
            generate_scenario,
            record_player_improv,
            generate_reaction,
            save_progress,
            final_summary,
        ])

# -------------------------
# Entrypoint
# -------------------------
def prewarm(proc: JobProcess):
    # prewarm VAD model for voice activity detection
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ud = UD()
    ud.state = ImprovState()
    ud.state.summary = ""

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(voice="en-US-matthew", style="Promo", text_pacing=True),
        vad=ctx.proc.userdata.get("vad"),
        turn_detection=MultilingualModel(),
        userdata=ud,
        preemptive_generation=True,
    )

    # store session for tools
    ud.session = session

    await ctx.connect()

    await session.start(
        agent=ImprovAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC()
        )
    )

    await session.say("Welcome to Improv Battle! What is your name?")
    await session.run()

# -------------------------
# Run worker
# -------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
