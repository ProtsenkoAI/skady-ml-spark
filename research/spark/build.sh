# The script to submit the app to cluster
# :param: -d: if provided, repacking dependencies to dependencies.zip (recommended to use if
# 	  new dependencies installed).
# :param: -s: if provided, recreating dist/ directory with updated sources, configs etc
#	  (recommended to use after every project update that affects workers)


repack_dependencies=false
repack_sources=true

while getopts ':ds:' OPTION; do
	case "$OPTION" in 
		d)
			repack_dependencies=true
		;;
		s)
			repack_sources=true
		;;
	esac
done

mkdir -p ./dist

if [ $repack_dependencies = true ]; then
	pipenv lock -r > requirements.txt
	pipenv run pip install -r requirements.txt -t ./dist/dependencies
	cd ./dist
	zip dependencies.zip -r ./dependencies/* > /dev/null 2>&1
	# deleting anything used for build
	rm -r dependencies/
	rm -f Pipfile.lock 
	cd ../
	rm requirements.txt
fi
echo "before, $repack_sources"
if [ $repack_sources = true ]; then
	cd ./dist
	echo "zipping"
	cp ../app/main.py ./main.py
	cp ../app/config.json ./config.json
	zip sources.zip -r ./../app/sources/* > /dev/null 2>&1
fi
