"""Write a Viggle Remake edit prompt (six-section format) with Claude Code (headless), looking at the frames."""
import subprocess, json, os, sys
import config
EXAMPLE = json.load(open(os.path.join(config.VIGGLE_DIR, "comfyui", "workflows", "remake_edit_api.json")))["11"]["inputs"]["prompt"]
def write(task, source_frames, ref_frames, pins, seconds, fps=24, out=None):
    """task: plain-English edit. source_frames: [(png, time_s)] stills of the ORIGINAL clip.
    ref_frames: [png] edited target keyframes, pinned at frame indices `pins`."""
    stamps = ", ".join(f"<Picture {i+1}> at 00:{p/fps:06.3f}" for i, p in enumerate(pins))
    files = "\n".join(f"- original clip at {t:.2f}s: {p}" for p, t in source_frames) + "\n" + "\n".join(f"- <Picture {i+1}> (edited target frame, pinned at frame {pins[i]}): {p}" for i, p in enumerate(ref_frames))
    ask = f"""You write prompts for Viggle Remake, a video-to-video edit model on MiniMax H3. Look at these image files with your Read tool:
{files}

The clip is {seconds:.3f} s at {fps} fps. The edit: {task}
The edited target frames are pixel-exact frames of the wanted result ({stamps}).

Write ONE prompt in exactly the format of the example below (sections subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music). Rules:
- Describe the real scene you see (setting, people, objects, light, camera move) concretely and literally.
- Define the edited object as a <Subject> with a precise description and say exactly where it is in the source and where it must be in the target (left/right third, on what surface, in front of or behind what).
- In retention_analysis say every other element is fully_preserved and copied frame for frame, including any person's motion; the edited subject is moved/removed as asked.
- In detailed_description, walk through the shot with timestamps and say at which timestamps the scene matches each <Picture N>.
- overall_soundscape: The soundtrack of <Video 1>, unchanged. non_diegetic_music: N/A
Output only the prompt text, nothing else.

EXAMPLE:
{EXAMPLE}"""
    r = subprocess.run([config.CLAUDE_BIN, "-p", ask, "--allowedTools", "Read", "--model", config.CLAUDE_MODEL], capture_output=True, text=True, timeout=600)
    txt = r.stdout.strip()
    if r.returncode != 0 or "subject_definitions" not in txt:
        raise RuntimeError(f"claude -p did not return a prompt (exit {r.returncode}): " + (r.stderr or txt)[-800:])
    txt = txt[txt.find("subject_definitions"):]
    if out: open(out, "w").write(txt)
    return txt
