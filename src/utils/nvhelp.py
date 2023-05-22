# pip install pynvml 
import pynvml
def nvidia_info():
    nvidia_dict = {
        "state": True,
        "nvidia_version": "",
        "nvidia_count": 0,
        "gpus": []
    }
    try:
        pynvml.nvmlInit()
        nvidia_dict["nvidia_version"] = pynvml.nvmlSystemGetDriverVersion()
        nvidia_dict["nvidia_count"] = pynvml.nvmlDeviceGetCount()
        for i in range(nvidia_dict["nvidia_count"]):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu = {
                "gpu_name": pynvml.nvmlDeviceGetName(handle),
                "total": memory_info.total,
                "free": memory_info.free,
                "used": memory_info.used,
                "temperature": f"{pynvml.nvmlDeviceGetTemperature(handle, 0)}℃",
                "powerStatus": pynvml.nvmlDeviceGetPowerState(handle)
            }
            nvidia_dict['gpus'].append(gpu)
    except pynvml.NVMLError as _:
        nvidia_dict["state"] = False
    except Exception as _:
        nvidia_dict["state"] = False
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass
    return nvidia_dict



def check_gpu_mem_usedRate():
    info = nvidia_info()
    # print(info)
    used = info['gpus'][0]['used']
    tot = info['gpus'][0]['total']
    print(f"GPU0 used: {used/(1024*1024*1024)}gb, tot: {tot/(1024*1024*1024)}gb, 使用率：{used/tot}")

