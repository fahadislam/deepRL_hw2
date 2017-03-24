gpuid=1
# for env in "SpaceInvadersDeterministic-v3" "BreakoutDeterministic-v3"
for env in "BreakoutDeterministic-v3"
do
    # ====================
    # normal: DQN
    # for type in "normal" "double-Q" "double-DQN" "duel" "linear" "linear-simple"
    for type in "normal" "double-Q" "double-DQN" "duel" "linear"
    do
        cachedir="cache/$env-$type"
        # echo $env, $type, $gpuid, $cachedir
        
        mkdir -p $cachedir
        echo "python dqn_atari.py --gpu $gpuid --type $type --env $env 2>&1 > $cachedir/run.log "
        
        let "gpuid += 1"
        let "gpuid %= 4"
    done
    # ====================    
done

           
