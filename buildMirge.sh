#!/bin/bash

# default branch for building mirgecom for this driver
#mirge_branch="test-tuo-perf-volpart"
mirge_branch="production-enable-duplication-checks"
# conda environment name
conda_env="mirgeDriver.Y3prediction"

usage()
{
  echo "Usage: $0 [options]"
  echo "  --use-ssh         Use ssh-keys to clone emirge/mirgecom"
  echo "  --restore-build   Build with previously stored version information for mirgecom and associated packages"
  echo "  --help            Print this help text."
}

opt_git_ssh=0
opt_restore_build=0

while [[ $# -gt 0 ]];do
  arg=$1
  shift
  case $arg in
    --use-ssh)
      opt_git_ssh=1
      ;;
    --restore-build)
      opt_restore_build=1
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "=== Error: unknown argument '$arg' ."
      usage
      exit 1
      ;;
  esac
done

git_method=""
if [ $opt_git_ssh -eq 1 ]; then
  git_method="--git-ssh"
fi

# get emirge
if [ -z "$(ls -A emirge)" ]; then
  if [ $opt_git_ssh -eq 1 ]; then
    echo "git clone git@github.com:illinois-ceesd/emirge.git emirge"
    git clone git@github.com:illinois-ceesd/emirge.git emirge
  else
    echo "git clone https://github.com/illinois-ceesd/emirge.git emirge"
    git clone https://github.com/illinois-ceesd/emirge.git emirge
  fi
else
  echo "emirge install already present. Remove to build anew"
fi

# install script for mirgecom, 
if [ ${opt_restore_build} -eq 1 ]; then
# attempt to restore an existing build
  echo "Building MIRGE-Com from existing known configuration"
  if [ ${MIRGE_PLATFORM} ]; then
    if [ -z "$(ls -A platforms/${MIRGE_PLATFORM})" ]; then
      echo "Unknown platform ${MIRGE_PLATFORM}"
      echo "Currently stored configurations are:"
      ls platforms
    else
      echo "Using version information for ${MIRGE_PLATFORM}"
      cd emirge

      versionDir="platforms/${MIRGE_PLATFORM}"
      if [ -z ${CONDA_PATH+x} ]; then
        echo "CONDA_PATH unset, installing new conda with emirge"
        echo "./install.sh --env-name=$conda_env $git_method --conda-env=${versionDir}/myenv.yml --pip-pkgs=${versionDir}/myreqs.txt"
        ./install.sh --env-name=$conda_env $git_method --conda-env=${versionDir}/myenv.yml --pip-pkgs=${versionDir}/myreqs.txt
      else
        echo "Using existing Conda installation, ${CONDA_PATH}"
        echo "./install.sh --conda-prefix=$CONDA_PATH --env-name=$conda_env --conda-env=${versionDir}/myenv.yml --pip-pkgs=${versionDir}/myreqs.txt"
        ./install.sh --conda-prefix=$CONDA_PATH --env-name=$conda_env --conda-env=${versionDir}/myenv.yml --pip-pkgs=${versionDir}/myreqs.txt
      fi
    fi
  else
    echo "Unknown platform. Set the environment variable MIRGE_PLATFORM for automated build and storage"
    echo "For example in bash: export MIRGE_PLATFORM=\"mac-m1\""
    echo "Automated build failed and will now exit."
  fi

else
# build with the current development head
  echo "Building MIRGE-Com from the current development head"
  echo "***WARNING*** may not be compatible with this driver ***WARNING"
  echo "Consider using --restore-build and setting MIRGE_PLATFORM as appropriate for this platform"

  cd emirge

  if [ -z ${CONDA_PATH+x} ]; then
    echo "CONDA_PATH unset, installing new conda with emirge"
    echo "./install.sh --env-name=${conda_env} ${git_method} --branch=${mirge_branch}"
    ./install.sh --env-name=${conda_env} ${git_method} --branch=${mirge_branch}
  else
    echo "Using existing Conda installation, ${CONDA_PATH}"
    echo "./install.sh --conda-prefix=$CONDA_PATH --env-name=${conda_env} ${git_method} --branch=${mirge_branch}"
    ./install.sh --conda-prefix=$CONDA_PATH --env-name=${conda_env} ${git_method} --branch=${mirge_branch}
  fi
fi

# add a few packages that are required for our development process
source config/activate_env.sh
#conda activate ${conda_env}
python -m pip install flake8 flake8-quotes pylint

# install the git hooks script to get linting on commits
cd ..
cp githooks/pre-commit .git/hooks/pre-commit

# install the prediction driver and utils
python -m pip install -e .
