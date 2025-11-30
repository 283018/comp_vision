#!/usr/bin/env fish

set -l script_name (basename (status --current-filename))
set nice_level -10
set xla_enabled 0

if test (id -u) -eq 0
    echo "Do not run as root to preserve env vars!"
    exit 1
end

set args $argv

if test (count $argv) -eq 0
    echo "Usage: $script_name [--xla] [--nice N] /path/to/script.py [args...]"
    echo "    --xla           enable XLA (default: disabled)"
    echo "    --nice <int>    set nice level (default: -10)"
    exit 1
end

while test (count $args) -gt 0
    switch $args[1]
        case '--xla'
            set xla_enabled 1
            if test (count $args) -gt 1
                set args $args[2..-1]
            else
                set args
            end
            continue
        case '--nice'
            if test (count $args) -lt 2
                echo "Missing value for --nice"
                echo "Usage: $argv[0] [--xla] [--nice N] /path/to/script.py [args...]"
                exit 1
            end
            set -l val $args[2]
            if not string match -qr '^-?[0-9]+$' -- $val
                echo "Invalid nice val: $val"
                exit 1
            end
            set nice_level $val
            if test (count $args) -gt 2
                set args $args[3..-1]
            else
                set args
            end
            continue
        case '--help'
            echo "Usage: $script_name [--xla] [--nice N] /path/to/script.py [args...]"
            echo "    --xla           enable XLA (default: disabled)"
            echo "    --nice <int>    set nice level (default: -10)"
            exit 1
        case '--*'
            echo "Unknown option: $args[1]"
            echo "Usage: $script_name [--xla] [--nice N] /path/to/script.py [args...]"
            echo "    --xla           enable XLA (default: disabled)"
            echo "    --nice <int>    set nice level (default: -10)"
            exit 1
        case '*'
            break
    end
end

if test (count $args) -eq 0
    echo "Usage: $script_name [--xla] [--nice N] /path/to/script.py [args...]"
    exit 1
end

set py $args[1]
set pyargs $args[2..-1]

set pydir (dirname -- $py)
set pybase (basename -- $py)

if test $xla_enabled -eq 1
    # set tf_xla_flag '--tf_xla_enable_xla_devices=true'
    set tf_xla_flag '--tf_xla_auto_jit=2'
else
    # set tf_xla_flag '--tf_xla_enable_xla_devices=false'
    set tf_xla_flag '--tf_xla_auto_jit=0'
end


sudo -v

#? #TODO: export TF_CUDNN_USE_AUTOTUNE=1 + export TF_DETERMINISTIC_OPS=0
set cmd "cd '$pydir' && \
    export TF_CPP_MIN_LOG_LEVEL=3 && \
    export GRPC_VERBOSITY=ERROR && \
    export GLOG_minloglevel=2 && \
    export ABSL_MIN_LOG_LEVEL=2 && \
    export GLOG_logtostderr=0 && \
    export TF_XLA_FLAGS='$tf_xla_flag' && \
    export TF_FORCE_GPU_ALLOW_GROWTH=true && \
    \
if [ -f 'venv/bin/activate' ]; then . 'venv/bin/activate'; \
elif [ -f '.venv/bin/activate' ]; then . '.venv/bin/activate'; \
fi && exec python3 \"\$@\""


# measuring here less accurate, but overall negligible
set -l start_hr (date '+%Y-%m-%d %H:%M:%S')
set -l start_ts (date +%s)

printf "\n\n\n"
printf "Starting at: %s\n" $start_hr
printf '%s\n' "##################################################"
printf "\n"

sudo --preserve-env=$PATH \
    systemd-run --scope \
        --expand-environment=yes \
        --nice=$nice_level \
        -p CPUWeight=10000 \
        -p IOWeight=10000 \
        bash -lc "$cmd" -- $pybase $pyargs

set -l py_exit $status

set -l end_hr (date '+%Y-%m-%d %H:%M:%S')
set -l end_ts (date +%s)

set -l delta (math $end_ts - $start_ts)
set -l hours (math "floor($delta / 3600)")
set -l minutes (math "floor(($delta % 3600) / 60)")
set -l seconds (math "$delta % 60")

printf "\n\n\n"
printf '\n%s\n' "##################################################"
printf "Finished at: %s\n" $end_hr
printf "Training time:\n"
printf "  %d seconds\n" $delta
printf "  %02d:%02d:%02d\n" $hours $minutes $seconds

exit $py_exit
