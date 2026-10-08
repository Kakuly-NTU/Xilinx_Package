import hal_py

def is_special_pin(pin: hal_py.ModulePin) -> bool:
    return pin.get_type() in (
        hal_py.PinType.power,
        hal_py.PinType.ground,
        hal_py.PinType.clock,
        hal_py.PinType.reset,
        hal_py.PinType.enable,
        hal_py.PinType.set,
    )

# Collect all nets to delete
def collect_nets_to_delete():
    nets_to_delete = []
    degree_sum = {
            hal_py.PinType.power: 0,
            hal_py.PinType.ground: 0,
            hal_py.PinType.clock: 0,
            hal_py.PinType.reset: 0,
            hal_py.PinType.enable: 0,
            hal_py.PinType.set: 0,
    }
    for net in netlist.get_nets():
        removed = False
        for src_ep in net.get_sources():
            if is_special_pin(src_ep.get_pin()):
                nets_to_delete.append(net)
                degree_sum[src_ep.get_pin().get_type()] += (
                        net.get_num_of_sources() + net.get_num_of_destinations()
                )
                removed = True
                break
        if removed:
            continue
        for dst_ep in net.get_destinations():
            if is_special_pin(dst_ep.get_pin()):
                nets_to_delete.append(net)
                degree_sum[dst_ep.get_pin().get_type()] += (
                        net.get_num_of_sources() + net.get_num_of_destinations()
                )
                removed = True
                break
    return nets_to_delete

if __name__ == "__main__":
    try:
        nets_to_delete_netlist = collect_nets_to_delete()
        print(f"Nets detected for removal: {len(nets_to_delete_netlist)}")
        for net in nets_to_delete_netlist:
            print("Deleting net:", net.get_name())
            netlist.delete_net(net)
    except Exception as e:
        print(f"Error while removing nets: {e}")
    