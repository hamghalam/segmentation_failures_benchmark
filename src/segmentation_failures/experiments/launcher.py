import argparse
import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import rich
from loguru import logger
from rich import print
from rich.syntax import Syntax

from segmentation_failures.experiments import filter_experiments, get_experiments
from segmentation_failures.experiments.cluster import submit
from segmentation_failures.experiments.experiment import EXPERIMENT_TASKS, Experiment

BASH_LOCAL_COMMAND = r"""
bash -c 'set -o pipefail; {command} |& tee -a "$HOME/outputs_segfail/{log_file_name}.log"'
"""

BASH_BASE_COMMAND = r"""
HYDRA_FULL_ERROR=1 python {project_root}/scripts/{task}.py {overwrites}
"""


async def worker(name, queue: asyncio.Queue[str]):
    while True:
        # Get a "work item" out of the queue.
        cmd = await queue.get()
        logger.info(f"{name} running {cmd}")
        proc = await asyncio.create_subprocess_shell(
            cmd,
        )

        # Wait for the subprocess exit.
        await proc.wait()

        if proc.returncode != 0:
            logger.error(f"{name} running {cmd} finished abnormally")
        else:
            logger.info(f"{name} running {cmd} finished")

        # Notify the queue that the "work item" has been processed.
        queue.task_done()


async def run(experiments_: list[Experiment], task: str, dry_run: bool, user_overwrites: dict):
    """Runs experiments in parallel.

    Args:
        experiments_ (list[Experiment]): experiments to run
        task (str): name of script (= task) to run
        dry_run (bool): dry run
        overwrites_dict (dict): overwrites coming from the command line
    """
    if len(experiments_) == 0:
        print("Nothing to run")
        return
    Path("~/outputs_segfail").expanduser().mkdir(exist_ok=True)

    # Create a queue that we will use to store our "workload".
    queue: asyncio.Queue[str] = asyncio.Queue()

    for experiment in experiments_:
        log_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{experiment.name}"
        # Compile overwrites. Precedence: cmdline > experiment
        final_overwrites = experiment.overwrites()
        final_overwrites.update(user_overwrites)
        final_overwrites = overwrites_to_str(final_overwrites)
        # escape special characters in overwrites
        overwrites = " ".join([f"'{k}={v}'" for k, v in final_overwrites.items()])

        cmd = BASH_BASE_COMMAND.format(
            project_root=Path(__file__).parents[1],
            task=task,
            overwrites=overwrites,
        ).strip()

        print(
            Syntax(
                re.sub(r"([^,]) ", "\\1 \\\n\t", cmd),
                "bash",
                word_wrap=True,
                background_color="default",
            )
        )

        cmd = BASH_LOCAL_COMMAND.format(command=cmd, log_file_name=log_file_name).strip()
        print(Syntax(cmd, "bash", word_wrap=True, background_color="default"))
        if not dry_run:
            queue.put_nowait(cmd)
        else:
            break

    if queue.empty():
        return

    async_tasks = [asyncio.create_task(worker("worker-0", queue))]
    # Wait until the queue is fully processed.
    await queue.join()

    # Cancel our worker tasks.
    for tsk in async_tasks:
        tsk.cancel()
    # Wait until all worker tasks are cancelled.
    await asyncio.gather(*async_tasks, return_exceptions=True)


def launch(
    task: str,
    dry_run: bool,
    cluster: bool,
    overwrites: list[str] | None,
    group_name: str = None,
    cpu: bool = False,
    gmem: str = "10.7G",
    **kwargs,
):
    """Launches experiments locally or on the cluster.

    See `add_arguments` for info on the arguments.
    """
    if overwrites is None:
        overwrites = {}
    experiment_list = get_experiments(task, group_name)
    # filter based on arguments
    experiment_list = filter_experiments(experiments=experiment_list, **kwargs)
    # user overwrites
    overwrites_dict = {}
    for x in overwrites:
        if "=" not in x:
            raise ValueError(f"Overwrite {x} must be formatted like key=value")
        k, v = x.split("=")
        overwrites_dict[k] = v
    print("Launching:")
    for exp in experiment_list:
        rich.print(f"{exp.name} (in group {exp.group if exp.group else 'default'})")
    if cluster:
        submit(experiment_list, task, dry_run, overwrites_dict, cpu=cpu, gmem=gmem)
    else:
        asyncio.run(run(experiment_list, task, dry_run, overwrites_dict))


def recursive_list_to_str(values: list) -> str:
    """Converts a list with nested lists to a string.
    This neglects types!
    """
    str_values = []
    for v in values:
        if isinstance(v, list):
            str_values.append(recursive_list_to_str(v))
        else:
            str_values.append(str(v))
    return "[" + ",".join(str_values) + "]"


def recursive_dict_to_str(values: dict) -> str:
    """Converts a dict with nested dicts to a string.
    This neglects types!
    """
    str_values = []
    for k, v in values.items():
        if isinstance(v, dict):
            str_values.append(f"{k}:{recursive_dict_to_str(v)}")
        else:
            str_values.append(f"{k}:{v}")
    return "{" + ",".join(str_values) + "}"


def overwrites_to_str(overwrites: dict[str, Any]) -> dict[str, str]:
    """Converts each overwrite value to a string.

    Hydra uses an override grammar that I try to follow here.
    Args:
        overwrites (dict[str, Any]): overwrites as a dict, which can contain lists and dicts

    Returns:
        dict[str, str]: overwrites as a dict of *strings*
    """
    str_overwrites = {}
    for k, v in overwrites.items():
        if isinstance(v, list):
            str_overwrites[k] = recursive_list_to_str(v)
        elif isinstance(v, dict):
            str_overwrites[k] = recursive_dict_to_str(v)
        else:
            str_overwrites[k] = str(v)
    return str_overwrites


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--task",
        choices=EXPERIMENT_TASKS,
        help="Task to perform in the experiment.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cluster", action="store_true", help="Whether to launch on the cluster")
    parser.add_argument(
        "--group",
        default=None,
        type=str,
        help="Allows grouping experiments. All experiments with the same group share the root directory.",
    )
    parser.add_argument(
        "--overwrites",
        type=str,
        nargs="+",
        required=False,
        help="Overwrites for config file. Syntax as usual in hydra: param1=value1 param2=value2 ...",
    )
    # cluster queue
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="For cluster: whether to use CPU cluster. Default is GPU cluster.",
    )
    parser.add_argument(
        "--gmem",
        default="10.7G",
        help="For cluster: GPU memory to allocate per job.",
    )
    return parser


def main():
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    # I chose to keep it simple here and not add every experiment attribute
    args, unknown_args = parser.parse_known_args()
    if len(unknown_args) % 2 != 0:
        raise ValueError("Unknown args must be in pairs")
    unknown_dict = {}
    for i in range(0, len(unknown_args), 2):
        unknown_dict[unknown_args[i].removeprefix("--")] = (
            unknown_args[i + 1] if unknown_args[i + 1] != "None" else None
        )
    launch(
        task=args.task,
        dry_run=args.dry_run,
        cluster=args.cluster,
        overwrites=args.overwrites,
        group_name=args.group,
        cpu=args.cpu,
        gmem=args.gmem,
        **unknown_dict,
    )


if __name__ == "__main__":
    main()
