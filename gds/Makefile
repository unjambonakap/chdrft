MAIN=map9v3
MAIN_V=${MAIN}.v
BUILD_DIR=./build
BUILD_MAIN_V=${BUILD_DIR}/${MAIN}.v
LEF_FILE=/home/benoit/opts/vlsi/share/qflow/tech/osu035/osu035_stdcells.lef
GDS_FILE=osu035_stdcells.gds2
MAKEFILE_BASE=./Makefile.base

clean: Makefile
	rm -r ${BUILD_DIR}

prepare: Makefile
	mkdir -p ${BUILD_DIR}
	cp ${MAIN_V} ${BUILD_DIR}/
	cp ${GDS_FILE} ${BUILD_DIR}/
	cp ${MAKEFILE_BASE} ${BUILD_DIR}/Makefile
	STDCELLS_GDS=${GDS_FILE} MAIN=${MAIN} LEF_FILE=${LEF_FILE} envsubst < ./genere_gds.sh.in > ${BUILD_DIR}/genere_gds.sh

build: prepare
	make -C ${BUILD_DIR} build

