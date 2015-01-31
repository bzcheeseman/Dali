for file in app.o stacked_lstm.o character_prediction.o sparkfun_prediction.o
do
	if [ -f $file ];
	then
		for lib in libmkl_intel_lp64.dylib libmkl_intel_thread.dylib libmkl_core.dylib 
		do
			install_name_tool -change $lib /opt/intel/mkl/lib/$lib $file
		done
    else
    	echo "install_name_tool can't find \"$file\""
    fi
done

