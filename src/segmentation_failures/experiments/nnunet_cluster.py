import re

from pssh.clients import SSHClient
from pssh.exceptions import Timeout
from rich import print
from rich.syntax import Syntax

BASH_BSUB_COMMAND = r"""
bsub -gpu num=1:j_exclusive=yes:gmem={gmem}\
    -L /bin/bash \
    -q gpu \
    -u 'm.zenk@dkfz-heidelberg.de' \
    -B -N \
    -R "select[hname!='e230-dgx2-2']" \
    -J "{name}" \
    -o $HOME/job_outputs/%J.out \
    bash -li -c 'set -o pipefail; echo $LSB_JOBID && source $HOME/.bashrc_nnunetV2 && {command}'
"""
# I use a .bashrc file here, which is easier on the cluster. dotenv won't override this


def get_gmem():
    return "10.7G"


def submit(nnunet_cmd: str, dry_run: bool):
    client = SSHClient("odcf-worker01.inet.dkfz-heidelberg.de")
    # Compile overwrites. Precedence: cmdline > experiment > global

    print(
        Syntax(
            re.sub(r"([^,]) ", "\\1 \\\n\t", nnunet_cmd),
            "bash",
            word_wrap=True,
            background_color="default",
        )
    )

    cmd = BASH_BSUB_COMMAND.format(
        name=nnunet_cmd,
        command=nnunet_cmd,
        gmem=get_gmem(),
    ).strip()

    print(
        Syntax(
            cmd,
            "bash",
            word_wrap=True,
            background_color="default",
        )
    )

    if dry_run:
        return
    with client.open_shell(read_timeout=1) as shell:
        shell.run(cmd)

        try:
            for line in shell.stdout:
                print(line)
        except Timeout:
            pass

        try:
            for line in shell.stderr:
                print(line)
        except Timeout:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nnunet_cmd", help="The nnUNet command to run. You need to pass it in quotes."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    submit(args.nnunet_cmd, args.dry_run)
