# Memoryless baselines on all six unseen types

```bash
.venv/bin/python -m alfworld.continual.memoryless_unseen \
  --reference-eval runs/frozen-heat200-to-clean-unseen-repeat1-20261126T025854Z/ascent \
  --output-dir runs/memoryless-all-unseen-seed42 \
  --device cuda:0
```

Queues one evaluation per task type, including heat&place, on the chosen GPU.
Model and inference parameters come from the reference transfer evaluation;
the full unseen manifest is shuffled once using the reference seed, then filtered
by task type. As in transfer evaluation, each task uses episode seed `42 + index`
when the reference seed is 42. No source adapter or conversation is loaded, and
no target history or parameter updates are retained. Room-exposure labels refer
only to the comparison heat200 run, not to memoryless training.

Use `--prepare-only` to inspect all six jobs without loading a model, or
`--target-task heat_place` (or another task name) to run just one type.
`launch.json` tracks the queue; each task directory contains the manifest,
provenance, per-episode results and summary. Use a fresh output directory.

# Retain ASCENT checkpoints at 50/100/150/200

```bash
.venv/bin/python -m alfworld.continual \
  --config configs/continual-heat-place-200-ascent-milestones.yaml \
  --output-dir runs/heat-place-200-ascent-milestones-seed42
```

This starts one fresh ASCENT run on the full 200 episodes of the original
heat200 seed42 manifest. Learning settings match the original heat200 ASCENT run.
`checkpoint_keep_positions: [50, 100, 150, 200]` saves each selected position after
that episode's update, including adapter, optimizer, RNG, and method state.
It forces a save even when the position falls between regular checkpoints.

`milestones/episode-0050.json` (and `0100`, `0150`, `0200`) records the position,
completed episode records, and the retained `.pkl` path relative to the run root.
The checkpoint is a hard link to immutable serialized state, so later cleanup of
rolling checkpoints does not remove it. Ordinary runs retain their existing
latest-checkpoint-only behavior. An interrupted run resumes with `--resume`.
These are new-run checkpoints, not recovered weights from the original run.
The new 200 checkpoint provides a comparison from the same training trajectory;
same-seed reruns need not reproduce the historical trajectory token for token.

# Frozen heat200 transfer

Evaluate the remaining four `valid_unseen` task types with the same ICL and
ASCENT sources used by an existing clean&place evaluation:

```bash
.venv/bin/python scripts/run_frozen_transfer_suite.py \
  --reference-eval runs/frozen-heat200-to-clean-unseen-repeat1-20261126T025854Z \
  --output-dir runs/frozen-heat200-to-remaining-unseen-repeat1 \
  --device cuda:5
```

Use `--prepare-only` to validate sources and inspect all eight jobs without loading
a model or creating output. The suite checks checkpoint hashes, ICL history hashes,
and matching target manifests before starting. It queues pick&place (24 episodes),
cool&place (21), look-at-light (18), and pick-two&place (17), each with ICL and
ASCENT. Counts refer to the local standard ALFWorld data after solvability filtering.
ICL keeps the source's final three complete episodes; ASCENT loads the final
adapter. Evaluation never updates weights or retains target episode history.

Jobs run sequentially on the selected device. `launch.json` records queued/running/
completed/failed status; each task/method directory contains its manifest,
provenance, per-episode results, and summary with room-exposure breakdowns. Failed
jobs do not prevent later jobs from running; the suite exits nonzero if any fail.
Use a fresh output directory; partial evaluations cannot be resumed.

For an individual evaluation, use `python -m alfworld.continual.transfer` with
`--source-run`, `--output-dir`, `--device`, and `--target-task` (`clean_place`,
`pick_place`, `cool_place`, `look_at_light`, or `pick_two_place`). Omitting the target
preserves the original clean&place protocol.

# Play


### TextWorld

Interact with TextWorld games:

```bash
$ alfworld-play-tw $ALFWORLD_DATA/json_2.1.1/train/pick_heat_then_place_in_recep-Potato-None-SinkBasin-14/trial_T20190908_231731_054988/ 
```

Use Tab to auto complete from a list of candidate actions from the text-engine.

```
-= Welcome to TextWorld, ALFRED! =-

You are in the middle of a room. Looking quickly around you, you see a sinkbasin 1,
a stoveburner 3, a cabinet 2, a microwave 1, a fridge 1, a cabinet 1, a cabinet 3,
a cabinet 7, a countertop 2, a cabinet 4, a stoveburner 4, a garbagecan 1, a cabinet 8,
a cabinet 6, a coffeemachine 1, a toaster 1, a countertop 1, a stoveburner 2,
a stoveburner 1, and a cabinet 5.

Your task is to: put a hot potato in sinkbasin.

> go to countertop 1
You arrive at loc 15. On the countertop 1, you see a plate 2, a spatula 3, a ladle 1,
a fork 3, a pan 1, a bread 1, a knife 1, a butterknife 1, a egg 3, a knife 2,
a spatula 2, a tomato 1, and a spatula 1.

> go to countertop 2
You arrive at loc 21. On the countertop 2, you see a saltshaker 1, a mug 2,
a spoon 1, a apple 1, a potato 2, a potato 1, a bowl 3, a cup 2, a fork 2,
a fork 1, and a apple 2.

> take potato 1 from countertop 2
You pick up the potato 1 from the countertop 2.

> go to microwave 1
You arrive at loc 10. The microwave 1 is closed.

> heat potato 1 with microwave 1
You heat the potato 1 using the microwave 1.

> go to sinkbasin 1
You arrive at loc 8. On the sinkbasin 1, you see a knife 3, a egg 2,
and a dishsponge 3.

> move potato 1 to sinkbasin 1
You won!
```

**Note**: When loading a TextWorld game, if you see the following error
```
  File "../python3.9/site-packages/textworld/logic/pddl_logic.py", line 123, in <listcomp>
    arguments = [Variable(get_var_name(arg), name2type[arg]) for arg in args if arg]
KeyError: 'val1'
```
it means the game is unsolvable or the PDDL problem is ill-defined. Some tasks from ALFRED like movable receptacle tasks are not used in ALFWorld, and will fail if you try to load them with `alfworld-play-tw`.

### THOR

Interact with embodied games:

```bash
$ alfworld-play-thor $ALFWORLD_DATA/json_2.1.1/train/pick_heat_then_place_in_recep-Potato-None-SinkBasin-14/trial_T20190908_231731_054988/ --controller oracle_astar --debug
```

Use `--agent mrcnn_astar` with `--debug` to inspect the BUTLER controller (MaskRCNN detector & A* Navigator).

![](../media/play_screenshot.png)
