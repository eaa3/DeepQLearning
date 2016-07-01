

os='unknown'
uname_out=$(uname)
if [[ "$uname_out" == "Linux" ]]; then
   os='linux'
elif [[ "$uname_out" == "Darwin" ]]; then
   os="osx"
fi


if [ ! -d dependencies ]; then
	echo "Dowloading Love game engine..."
	mkdir dependencies
fi

cd dependencies

if [ ! -d love.app ]; then

	if [[ "$os" == "osx" ]]; then


		wget https://bitbucket.org/rude/love/downloads/love-0.10.1-macosx-x64.zip
		tar -xf love-0.10.1-macosx-x64.zip
		rm *.zip

		echo "Setting up alias for engine application..."
		LOVE_PATH=$(pwd)

		alias love=$LOVE_PATH/love.app/Contents/MacOS/love
		#echo "alias love=$LOVE_PATH/love.app/Contents/MacOS/love" >> $HOME/.profile
	elif [[ "$os" == "linux" ]]; then
		echo "TODO"
	fi

fi

cd ..


