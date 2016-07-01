
echo "Dowloading Love game engine..."

rm -r dependencies 2> /dev/null
mkdir dependencies 2> /dev/null
cd dependencies

wget https://bitbucket.org/rude/love/downloads/love-0.10.1-macosx-x64.zip
tar -xf love-0.10.1-macosx-x64.zip
rm *.zip


echo "Setting up alias for engine application..."

LOVE_PATH=$(pwd)
alias love=$LOVE_PATH/love.app/Contents/MacOS/love

cd ..
