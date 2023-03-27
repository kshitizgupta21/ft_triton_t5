cd FasterTransformer
cd build 
git config --global --add safe.directory /workspace/FasterTransformer/build/_deps/googletest-src
git config --global --add safe.directory /workspace/FasterTransformer
git config --global --add safe.directory /workspace/FasterTransformer/3rdparty/cutlass
cmake -DSM=80,86 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON ..
make -j${nproc}
