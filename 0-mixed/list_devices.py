from tensorflow.python.client import device_lib

def get_available_devices():
    #nota: e' un metodo in sviluppo. potrebbe cambiare nome e smettere di funzionare
    local_device_protos = device_lib.list_local_devices()
    return [x.device_type + " " + x.name for x in local_device_protos if x.device_type == 'CPU']

print get_available_devices()