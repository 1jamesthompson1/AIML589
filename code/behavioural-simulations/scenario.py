"""Single-scenario entry point (one profile x one situation).

The ``scenario`` task lives in its own file because ``inspect eval`` on a
task file instantiates EVERY ``@task`` in it: a task with required arguments
(``profile`` / ``situation``) would crash a whole-file run of
``run_simulations.py`` before any task starts. Keeping it here leaves
``run_simulations.py`` runnable as a whole.

    inspect eval scenario.py \\
        -T profile=welfare -T situation=overpayment_recovery \\
        --model openai/Qwen/Qwen3.6-27B
"""

from inspect_ai import Task, task, task_with

from run_simulations import _profile_samples, profile_task


@task
def scenario(
    profile: str,
    situation: str,
    message_limit: int | None = None,
) -> Task:
    """A single scenario (one profile × one situation).

    Derived from the profile's task with ``task_with()`` (the documented task
    reuse pattern): the base task is built fresh on every call because
    ``task_with()`` mutates in place; the dataset is narrowed to the one
    situation, and limits can be overridden.

    Args:
        profile: Profile id (a directory in ``profiles/``).
        situation: Situation id (an entry in that profile's ``situations.json``).
        message_limit: Optional message limit override for the task.
    """
    samples = _profile_samples(profile, situation)
    if len(samples) != 1:
        raise ValueError(
            f"expected one situation {situation!r} for profile {profile!r}"
        )

    overrides: dict = {
        "dataset": samples,
        "name": f"wvs-simulations-{profile}-{situation}",
        "metadata": {
            "profile": profile,
            "situation": situation,
            "derived_from": f"wvs-simulations-{profile}",
        },
    }
    if message_limit is not None:
        overrides["message_limit"] = message_limit
    return task_with(profile_task(profile), **overrides)
