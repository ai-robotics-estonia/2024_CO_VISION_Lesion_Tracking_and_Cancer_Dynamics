while getopts t:g:n: flag
do
    case "${flag}" in
        t) request_time=${OPTARG};;
        g) request_gpu=${OPTARG};;
        n) num_gpu=${OPTARG};;
    esac
done

srun --pty --time=$request_time --mem=64000 --partition=gpu --gres=gpu:$request_gpu:$num_gpu --cpus-per-task=8 -A revvity --exclude=falcon3 bash