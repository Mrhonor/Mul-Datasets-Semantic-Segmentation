# some more ls aliases
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

# default file mode 0666
umask 0000

PS1="\w> "

export CUDA_HOME=/usr/local/cuda-11.1:$CUDA_HOME
export PATH=/usr/local/cuda-11.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/cpfs01/projects-HDD/pujianxiangmuzu_HDD/mr_22210240239/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cpfs01/projects-HDD/pujianxiangmuzu_HDD/mr_22210240239/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/cpfs01/projects-HDD/pujianxiangmuzu_HDD/mr_22210240239/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/cpfs01/projects-HDD/pujianxiangmuzu_HDD/mr_22210240239/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

