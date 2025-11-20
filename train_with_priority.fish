#!/usr/bin/env fish

# TODO: enable for changeable input??
# set -x TF_XLA_FLAGS "--tf_xla_enable_xla_devices=false"

# disable annoying tf warning on import
set -x TF_CPP_MIN_LOG_LEVEL 3
set -x GLOG_minloglevel 2
set -x GLOG_logtostderr 1
set -x GRPC_VERBOSITY ERROR
set -x ABSL_MIN_LOG_LEVEL 2

set script $argv[1]
set rest_args
if test (count $argv) -gt 1
    set rest_args $argv[2..-1]
end

# idk, should preserve vram at startup
set -x TF_FORCE_GPU_ALLOW_GROWTH true

if test (id -u) -eq 0
    set NICE_CMD "nice -n -5"
else
    set NICE_CMD "nice -n 0"
    printc --yellow "Running as non-root; renice to negative"
end

set -l inner_cmd "$NICE_CMD ionice -c2 -n0 python3 -u -- '$script' $rest_args"

if type -q systemd-run
    set unitname "train-"(date +%s)
    printc --green "Starting: unit: $unitname, CPUWeight=1000 ..."
    systemd-run --scope -p CPUWeight=1000 -p IOWeight=1000 --unit=$unitname bash -lc "$inner_cmd"
    exit 0
end

# if no systemd-run
printc --yellow "systemd-run not available - starting directly"
eval $inner_cmd &

set pid $last_pid
printc --green "started PID $pid"

set pid (pgrep -f $script)
renice -n -5 -p pid
