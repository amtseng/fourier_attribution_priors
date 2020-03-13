import os
import json
import sys
import tqdm

def find_broken_metrics(path, json_name="metrics.json", verbose=True):
    """
    From a base directory, recursively finds the path to all JSONs matching
    the given JSON name, that is ill-formed.
    """
    if verbose:
        print(path)
    broken = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if item == json_name:
            try:
                with open(item_path, "r") as f:
                    json.load(f)
            except json.JSONDecodeError:
                broken.append(item_path)
        elif os.path.isdir(item_path):
            broken.extend(find_broken_metrics(item_path))
    return broken


def fix_broken_metrics(json_path):
    """
    Attempts to fix an ill-formed JSON by adding any missing closing ] or } at
    the end. This can only fix ill-formed JSONs that are missing ] or } a the
    end, and this assumes that the JSON does not have any escaped [ or {.
    """
    openers = []
    with open(json_path, "r") as f:
        json_str = f.read()
        for char in json_str:
            if char == "}":
                if openers[-1] == "{":
                    openers.pop()
                else:
                    raise ValueError("Found } with no { first")
            elif char == "{":
                openers.append(char)
            if char == "]":
                if openers[-1] == "[":
                    openers.pop()
                else:
                    raise ValueError("Found ] with no [ first")
            elif char == "[":
                openers.append(char)
    if not openers:
        raise ValueError("Did not find missing } or ] at end")
    while openers:
        char = openers.pop()
        if char == "{":
            json_str += "}"
        else:
            json_str += "]"
    try:
        obj = json.loads(json_str)  # Try to load the JSON
        with open(json_path, "w") as f:
            json.dump(obj, f, sort_keys=True, indent=2)
    except json.JSONDecodeError:
        raise ValueError("Fix didn't work")


if __name__ == "__main__":
    base_path = sys.argv[1] if len(sys.argv) > 1 else "."
    broken = find_broken_metrics(base_path)

    if not broken:
        print("No broken metrics.json files found")
        sys.exit(0)

    print("Found the following broken metrics.json files:")
    print("----------------------------------------------")
    for path in broken:
        print(path)

    resp = input("Attempt to fix? [y/N]: ")
    if resp and resp[0].lower() == "y":
        for path in tqdm.tqdm(broken, desc="Fixing..."):
            try:
                fix_broken_metrics(path)
            except ValueError as e:
                print("Could not fix %s: %s" % (path, e.args[0]))
    print("Done")
