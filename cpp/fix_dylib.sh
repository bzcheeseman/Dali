INTEL_THREAD_FILE=/opt/intel/mkl/lib/libmkl_intel_thread.dylib
IOMP5_FILE=/opt/intel/composer_xe_2015.1.108/compiler/lib/libiomp5.dylib
for file in *.o
do
	echo "Fixing \"$file\""
	if [ -f $file ];
	then
		for lib in libmkl_intel_lp64.dylib libmkl_intel_thread.dylib libmkl_core.dylib 
		do
			install_name_tool -change $lib /opt/intel/mkl/lib/$lib $file
		done
		install_name_tool -change libiomp5.dylib $IOMP5_FILE $file
    else
    	echo "install_name_tool can't find \"$file\""
    fi
done
if [ -w "$INTEL_THREAD_FILE" ]
then
	install_name_tool -change libiomp5.dylib $IOMP5_FILE $INTEL_THREAD_FILE
    echo "Achievement Unlocked !!"
    echo ""
    echo "Linking fixed on all files"
else
	echo "Could not fix linking for $INTEL_THREAD_FILE rerun with 'sudo'"
fi
