#!/bin/bash
apt-get update -y

apt-get install screen -y
if !(ls -al ~/ | grep ".screenrc"); then
	echo 'ck 5000' >> ~/.screenrc
	echo 'vbell off' >> ~/.screenrc
	echo 'defscrollback 10000' >> ~/.screenrc
	echo 'termcapinfo xterm* ti@:te@' >> ~/.screenrc
	echo 'startup_message off' >> ~/.screenrc
	echo 'hardstatus on' >> ~/.screenrc
	echo 'hardstatus alwayslastline' >> ~/.screenrc
	echo 'hardstatus string "%{.bW}%-w%{.rW}%n*%t%{-}%+w %= %c ${USER}@%H"' >> ~/.screenrc
	echo 'bindkey -k k1 select 0' >> ~/.screenrc
	echo 'bindkey -k k2 select 1' >> ~/.screenrc
	echo 'bindkey -k k3 select 2' >> ~/.screenrc
fi

apt-get install language-pack-ko -y

if !(grep -qc "LANG" ~/.bashrc); then
	echo 'export LANG="ko_KR.UTF-8"' >> ~/.bashrc
fi

apt-get install curl -y
apt-get install build-essential checkinstall -y
apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev -y
apt-get install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt-get install python3.11 python3.11-dev

curl -sSL https://install.python-poetry.org | python3 -

if !(grep -qc "/opt/ml/.local/bin" ~/.bashrc); then
	echo 'export PATH="/opt/ml/.local/bin:$PATH"' >> ~/.bashrc
fi
