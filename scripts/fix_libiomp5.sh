INTEL_THREAD_FILE=/opt/intel/mkl/lib/libmkl_intel_thread.dylib
IOMP5_FILE=/opt/intel/composer_xe_2015.1.108/compiler/lib/libiomp5.dylib
red=`tput setaf 1`
green=`tput setaf 2`
reset=`tput sgr0`

if [ -w "$INTEL_THREAD_FILE" ]
then
    install_name_tool -change libiomp5.dylib $IOMP5_FILE $INTEL_THREAD_FILE
    echo "${green}"
    echo "    ┌───────────────────────┐"
    echo "    │Achievement Unlocked !!│"
    echo "    └───────────────────────┘"
    echo "${reset}"
    echo " -> Linking fixed on all files"
else
    echo "${red}"
    echo "    ┌────────┐"
    echo "    │FAILURE!│"
    echo "    └────────┘"
    echo "${reset}"
    echo " -> Could not fix linking for $INTEL_THREAD_FILE"
    echo "    rerun with 'sudo'"
fi