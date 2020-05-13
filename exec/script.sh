for condition in allSpeed0.8 allSpeed1 allSpeed1.2 AnySpeed0.8 AnySpeed1
do
    cd ~/Downloads/goalCommitment-project/multiAgentChasing/data/${condition}

    ffmpeg -r  30 -f image2 -s 1920x1080 -i  %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p ~/Downloads/goalCommitment-project/multiAgentChasing/demo/${condition}_30.mp4

    cd ~/Downloads/goalCommitment-project/multiAgentChasing/exec
done