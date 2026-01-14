def print_config(cfg):
    print("Current config:")
    for k, v in cfg.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:   
            print(f"  {k}: {v}")
    print()