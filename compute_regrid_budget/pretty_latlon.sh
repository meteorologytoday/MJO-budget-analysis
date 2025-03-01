#!/bin/bash

mymod () {

    local x="$1"
    local m="$2"

    local res=$(( x % m ))

    if (( x < 0 )); then
        res=$(( res + m ))
    fi

    echo $res
}


pretty_lat () {

    local lat="$1"
    local suffix=""

    if (( lat > 0 )); then
        suffix="N"
    elif (( lat < 0 )); then
        suffix="S"
        lat=$(( - lat ))
    else
        suffix="E"
    fi

    echo $( printf "%d%s" $lat $suffix )
}

pretty_lon () {

    local lon=$( mymod "$1" 360 )
    local suffix=""

    if (( lon <= 180 )); then
        suffix="E"
    else
        suffix="W"
        lon=$(( 360 - lon ))
    fi

    echo $( printf "%d%s" $lon $suffix )
}

