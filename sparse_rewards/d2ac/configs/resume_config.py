import json
import os.path as osp


def check_resume_config(resume_folder, args):
    # read the `config.json` file from the resume_path folder
    # and check that the config is the same as the current config in args
    json_path = osp.join(resume_folder, "config.json")
    # read the json file
    json_file = open(json_path, "r")
    json_dict = json.load(json_file)
    json_file.close()
    # compare the json_dict with args
    args_dict = vars(args)
    keys_not_matched = []
    for key in json_dict:
        if key in args_dict:
            if json_dict[key] != args_dict[key]:
                print(
                    f"Value for key {key} differs. In {json_path}: {json_dict[key]}, in args: {args_dict[key]}"
                )
                keys_not_matched.append(key)
        else:
            print(f"Key {key} in {json_path} is not present in args.")
            keys_not_matched.append(key)

    if set(keys_not_matched) == set(["seed", "exp_name", "resume_path"]):
        print("All keys matched.")
    else:
        print(
            "Not all keys matched; check: ",
            set(keys_not_matched) - set(["seed", "exp_name", "resume_path"]),
        )
