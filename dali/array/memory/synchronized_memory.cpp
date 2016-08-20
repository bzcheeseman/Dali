#include "dali/array/memory/synchronized_memory.h"

#include <cassert>

#include "dali/array/memory/memory_bank.h"
#include "dali/utils/print_utils.h"

namespace memory {
    ////////////////////////////////////////////////////////////////////////////////
    //                           DEVICE MEMORY                                    //
    ////////////////////////////////////////////////////////////////////////////////

    SynchronizedMemory::DeviceMemory::DeviceMemory() : ptr(NULL), allocated(false), fresh(false) {}

    ////////////////////////////////////////////////////////////////////////////////
    //                     SYNCHRONIZED MEMORY                                    //
    ////////////////////////////////////////////////////////////////////////////////

    SynchronizedMemory::DeviceMemory& SynchronizedMemory::get_device_memory(Device device) const {
        if (device.is_cpu()) {
            return cpu_memory;
        }
#ifdef DALI_USE_CUDA
        else if (device.is_gpu()) {
            return gpu_memories[device.number()];
        }
#endif
        else if (debug::enable_fake_devices && device.is_fake()) {
            return debug::fake_device_memories[device.number()];
        } else {
            ASSERT2(false, "Unsupported device passed to SynchronizedMemory.");
            return cpu_memory;
        }
    }

    void SynchronizedMemory::iterate_device_memories(device_iterator_t f) const {
        f(Device::cpu(), cpu_memory);
#ifdef DALI_USE_CUDA
        for (int i=0; i<DALI_MAX_GPU_DEVICES;++i) {
            f(Device::gpu(i), gpu_memories[i]);
        }
#endif
        if (debug::enable_fake_devices) {
            for (int i=0; i<debug::MAX_FAKE_DEVICES; ++i) {
                f(Device::fake(i), debug::fake_device_memories[i]);
            }
        }
    }

    void SynchronizedMemory::adopt_buffer(const Device& device, void* buffer) {
        // mark memory as about to be replaced by buffer
        auto& dev_memory = get_device_memory(device);
        free_device_memory(device, dev_memory);
        // notify memory that a new buffer has taken over
        mark_all_not_fresh();
        // assign the pointer and make sure that the state
        // reflects that this is now the freshest memory around
        dev_memory.ptr = buffer;
        dev_memory.allocated = true;
        dev_memory.fresh = true;
    }

    void SynchronizedMemory::mark_fresh(const Device& device) {
        get_device_memory(device).fresh = true;
    }

    void SynchronizedMemory::mark_all_not_fresh() {
        iterate_device_memories([](const Device& device, DeviceMemory& device_memory) {
            device_memory.fresh = false;
        });
    }

    void SynchronizedMemory::free_device_memory(const Device& device, SynchronizedMemory::DeviceMemory& dev_memory) const {
        if (dev_memory.allocated) {
            memory::bank::deposit(DevicePtr(device, dev_memory.ptr), total_memory, inner_dimension);
            dev_memory.ptr = (void*)NULL;
            dev_memory.allocated = false;
        }
    }

    SynchronizedMemory::SynchronizedMemory(int _total_memory,
                                           int _inner_dimension,
                                           Device _preferred_device,
                                           bool _clear_on_allocation) :
            total_memory(_total_memory),
            inner_dimension(_inner_dimension),
            preferred_device(_preferred_device),
            clear_on_allocation(_clear_on_allocation) {
        assert(total_memory % inner_dimension == 0);
    }

    SynchronizedMemory::SynchronizedMemory(const SynchronizedMemory& other) :
            SynchronizedMemory(other.total_memory, other.inner_dimension, other.preferred_device, other.clear_on_allocation) {


        allocate(preferred_device);
        auto dest = DevicePtr(preferred_device, get_device_memory(preferred_device).ptr);

        // find a source device, such that the copying process will have the least overhead.
        Device best_device = Device::device_of_doom();
        bool device_type_matching = false;
        bool device_number_match = false;

        other.iterate_device_memories([this,&best_device,&device_type_matching,&device_number_match]
                                      (const Device& device, DeviceMemory& device_memory) {
            if (device_memory.fresh) {
                bool current_device_type_matching   = device.type()   == this->preferred_device.type();
                bool current_device_number_matching = device.number() == this->preferred_device.number();

                bool update = false;
                if (best_device == Device::device_of_doom()) {
                    update = true;
                } else if (!device_type_matching && current_device_type_matching) {
                    update = true;
                } else if (!device_number_match && current_device_number_matching) {
                    update = true;
                }

                if (update) {
                    best_device          = device;
                    device_type_matching = current_device_type_matching;
                    device_number_match  = current_device_number_matching;
                }
            }
        });

        // TODO(szymon): if we do this with two different gpus then death.

        if (best_device == Device::device_of_doom()) return;

        auto source = DevicePtr(best_device, other.get_device_memory(best_device).ptr);

        memory::copy(dest, source, total_memory, inner_dimension);
        get_device_memory(preferred_device).fresh = true;
    }

    SynchronizedMemory::~SynchronizedMemory() {
        iterate_device_memories([this](const Device& device, DeviceMemory& device_memory) {
            free_device_memory(device, device_memory);
        });
    }

    bool SynchronizedMemory::is_fresh(const Device& device) const {
        return get_device_memory(device).fresh;
    }

    bool SynchronizedMemory::is_allocated(const Device& device) const {
        return get_device_memory(device).allocated;
    }

    memory::Device SynchronizedMemory::find_some_fresh_device() const {
        auto result = memory::Device::device_of_doom();
        iterate_device_memories([&result](const Device& device, DeviceMemory& device_memory) {
            if (device_memory.fresh) {
                result = device;
            }
        });
        return result;
    }

    bool SynchronizedMemory::is_any_fresh() const {
        bool result = false;
        iterate_device_memories([&result](const Device& device, DeviceMemory& device_memory) {
            if (device_memory.fresh) {
                result = true;
            }
        });
        return result;
    }

    bool SynchronizedMemory::is_any_allocated() const {
        bool result = false;
        iterate_device_memories([&result](const Device& device, DeviceMemory& device_memory) {
            if (device_memory.allocated) {
                result = true;
            }
        });
        return result;
    }

    void SynchronizedMemory::clear() {
        clear_on_allocation = true;

        this->allocate(preferred_device);
        auto ptr = get_device_memory(preferred_device).ptr;
        memory::clear(DevicePtr(preferred_device, ptr), total_memory, inner_dimension);

        mark_all_not_fresh();
        mark_fresh(preferred_device);
    }


    void SynchronizedMemory::lazy_clear() {
        clear_on_allocation = true;

        mark_all_not_fresh();

        // clear immediately if some memory was already allocated.
        bool some_allocated = false;
        iterate_device_memories([&some_allocated](const Device& device, DeviceMemory& device_memory) {
            if (device_memory.allocated) {
                some_allocated = true;
            }
        });
        if (some_allocated) {
            clear();
        }

    }

    bool SynchronizedMemory::allocate(const Device& device) const {
        auto& dev_memory = get_device_memory(device);
        if (dev_memory.allocated) {
            return false;
        }

        auto dev_ptr = memory::bank::allocate(device, total_memory, inner_dimension);
        dev_memory.ptr = dev_ptr.ptr;
        dev_memory.allocated = true;
        return true;
    }


    void SynchronizedMemory::free(const Device& device) const {
        auto& dev_memory = get_device_memory(device);
        free_device_memory(device, dev_memory);
    }

    void SynchronizedMemory::disown_buffer(const Device& device) const {
        auto& dev_memory = get_device_memory(device);
        dev_memory.ptr = NULL;
        dev_memory.allocated = false;
        dev_memory.fresh = false;
    }

    void SynchronizedMemory::move_to(const Device& target_device) const {
        auto& target_memory = get_device_memory(target_device);

        // if memory is fresh, we are done
        if (target_memory.fresh) {
            return;
        }

        // make sure memory is allocated.
        bool just_allocated = allocate(target_device);

        // if another piece of memory is fresh copy from it.
        iterate_device_memories([this, &target_device, &target_memory]
                                (const Device& source_device, DeviceMemory& source_device_memory) {
            // important to check !target_memory.fresh, not to load fresh memory
            // multiple times in case of more than one fresh devices.
            if (!target_memory.fresh && source_device_memory.fresh) {
                auto source = DevicePtr(source_device, source_device_memory.ptr);
                auto dest   = DevicePtr(target_device, target_memory.ptr);
                memory::copy(dest, source, total_memory, inner_dimension);
                target_memory.fresh = true;
            }
        });

        // if just allocated, and we need to clear the memory.
        // this only happens if we did not copy memory from elsewhere.
        if (just_allocated && clear_on_allocation && !target_memory.fresh) {
            memory::clear(DevicePtr(target_device, target_memory.ptr), total_memory, inner_dimension);
        }
        // even if there was no fresh copy, then we just
        // allocated and this copy becomes the fresh one.
        target_memory.fresh = true;
    }

    void SynchronizedMemory::to_cpu() const {
        move_to(memory::Device::cpu());
    }

#ifdef DALI_USE_CUDA
    void SynchronizedMemory::to_gpu(const int& gpu_number) const {
        move_to(memory::Device::gpu(gpu_number));
    }
#endif

    void* SynchronizedMemory::data(const Device& device, AM access_mode) {
        if (access_mode == AM_READONLY) {
            return readonly_data(device);
        } else if (access_mode == AM_MUTABLE) {
            return mutable_data(device);
        } else if (access_mode == AM_OVERWRITE) {
            return overwrite_data(device);
        } else {
            ASSERT2(false, "Unsupported access mode passed to SynchronizedMemory::data.");
            return NULL;
        }
    }

    void* SynchronizedMemory::readonly_data(const Device& device) const {
        move_to(device);
        return get_device_memory(device).ptr;
    }

    void* SynchronizedMemory::mutable_data(const Device& device) {
        move_to(device);
        mark_all_not_fresh();
        mark_fresh(device);
        return get_device_memory(device).ptr;
    }

    void* SynchronizedMemory::overwrite_data(const Device& device) {
        allocate(device);
        mark_all_not_fresh();
        mark_fresh(device);

        return get_device_memory(device).ptr;
    }

    void SynchronizedMemory::debug_info(std::basic_ostream<char>& stream,
                                        bool print_contents,
                                        DType dtype) const {
        stream << "Synchronized Memory (" << this << ")" << std::endl;
        stream << "    total_memory: " << total_memory << " bytes" << std::endl;
        stream << "    inner_dimension: " << inner_dimension << " bytes" << std::endl;
        stream << "    preferred_device: " << preferred_device.description() << std::endl;
        stream << "    clear_on_allocation: " << clear_on_allocation << std::endl;
        for (auto device: Device::installed_devices()) {
            stream << "    Device " << device.description(true) << std::endl;
            stream << "        fresh: " << get_device_memory(device).fresh << std::endl;
            stream << "        allocated: " << get_device_memory(device).allocated << std::endl;
            stream << "        ptr: " << get_device_memory(device).ptr << std::endl;
            if(print_contents && device.is_cpu() && get_device_memory(device).allocated) {
                stream << "        contents: [";
                if (dtype == DTYPE_FLOAT) {
                    float* ptr = (float*)get_device_memory(device).ptr;
                    for (int i=0; i<total_memory / sizeof(float); ++i) {
                        stream << *ptr << " ";
                        ptr++;
                    }
                } else if (dtype == DTYPE_DOUBLE) {
                    double* ptr = (double*)get_device_memory(device).ptr;
                    for (int i=0; i<total_memory / sizeof(double); ++i) {
                        stream << *ptr << " ";
                        ptr++;
                    }
                } else if (dtype == DTYPE_INT32) {
                    int* ptr = (int*)get_device_memory(device).ptr;
                    for (int i=0; i<total_memory / sizeof(int); ++i) {
                        stream << *ptr << " ";
                        ptr++;
                    }
                }
                stream << "]" << std::endl;
            }
        }
    }

    namespace debug {
        SynchronizedMemory::DeviceMemory fake_device_memories[MAX_FAKE_DEVICES];
    }
}
