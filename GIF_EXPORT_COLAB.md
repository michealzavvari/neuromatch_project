# Export a Rollout GIF (Recommended: Google Colab / Linux)

## Why this exists
On macOS, the current renderer in this repo uses legacy OpenGL calls (`glPushMatrix`, `glBegin`, etc.).  
macOS often provides an OpenGL Core Profile context where these calls are invalid, so rendering may crash.

If your end goal is a **GIF** (not realtime visualization), the most reliable workflow is:
**run evaluation + frame capture on Linux (Google Colab or a Linux VM)** and export a GIF.

---

## Option A: Google Colab (fastest)

### Step 1 — Get the project into Colab
Choose one:

**A) Clone from GitHub**
```bash
!git clone <YOUR_REPO_URL>
%cd neuromatch_project
```

**B) Upload a ZIP**
1. Upload your project zip to Colab.
2. Unzip it:
```bash
!unzip -q Archive.zip -d neuromatch_project
%cd neuromatch_project
```

---

### Step 2 — Install dependencies
```bash
!pip install "pyglet<2" imageio
```

If Gym is required by your project:
```bash
!pip install gym==0.21.0
```

---

### Step 3 — Create the GIF exporter script
Create a file named `make_gif.py` in the project root with the content below.

```python
import os
import argparse
import imageio.v2 as imageio
import torch

from multiagent import scenarios
from multiagent.environment import MultiAgentEnv
from MADDPG import MADDPG


def make_env(env_name: str):
    scenario = scenarios.load(f"{env_name}.py").Scenario()
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def load_maddpg(model_dir: str, env):
    obs_dim_list = [obs_space.shape[0] for obs_space in env.observation_space]
    act_dim_list = [act_space.n for act_space in env.action_space]
    maddpg = MADDPG(obs_dim_list, act_dim_list, 0, 0, 0)

    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pt not found at: {model_path}")

    data = torch.load(model_path, map_location="cpu")
    for agent, actor_parameter in zip(maddpg.agents, data):
        agent.actor.load_state_dict(actor_parameter)

    return maddpg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True)
    p.add_argument("--folder", required=True)
    p.add_argument("--episode-length", type=int, default=100)
    p.add_argument("--gif-path", default=None)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    model_dir = os.path.join("results", args.env, str(args.folder))
    os.makedirs(model_dir, exist_ok=True)

    env = make_env(args.env)
    maddpg = load_maddpg(model_dir, env)

    obs = env.reset()
    frames = []

    for _ in range(args.episode_length):
        actions = maddpg.select_action(obs)
        obs, rewards, dones, infos = env.step(actions)

        # NOTE: shared_viewer=True => env.render returns a list with 1 frame
        out = env.render(mode="rgb_array")
        frame = out[0] if isinstance(out, list) and len(out) else out
        if frame is not None:
            frames.append(frame)

    gif_path = args.gif_path or os.path.join(
        model_dir, f"rollout_{args.env}_folder{args.folder}.gif"
    )
    imageio.mimsave(gif_path, frames, fps=args.fps)
    print("Saved GIF:", gif_path)


if __name__ == "__main__":
    main()
```

---

### Step 4 — Run and produce a GIF
Example:
```bash
!python make_gif.py --env simple_adversary --folder 1 --episode-length 100 --fps 30
```

Output file:
```
results/<env>/<folder>/rollout_<env>_folder<folder>.gif
```

---

## Option B: Linux VM / Remote Linux
The same `make_gif.py` workflow works on any Linux machine:

1) Create a venv  
2) Install deps:
```bash
pip install "pyglet<2" imageio
```

3) Run:
```bash
python make_gif.py --env <ENV_NAME> --folder <FOLDER> --episode-length 100 --fps 30
```

---

## Notes
- This exports a single episode rollout as a GIF.
- If you want multiple episodes, run the script multiple times (or extend it with an outer episode loop).
