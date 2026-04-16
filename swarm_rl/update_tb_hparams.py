from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.plugins.hparams import plugin_data_pb2
from tensorboard.compat.tensorflow_stub.io.gfile import GFile
from tensorboard.plugins.hparams.plugin_data_pb2 import HParamsPluginData
# import zlib
from global_cfg import QuadrotorEnvConfig
import sys

import struct, os, copy

import crc32c
def _masked_crc32c(data):
    crc = crc32c.crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xa282ead8) & 0xffffffff

def _write_events(path, events):
    with open(path, "wb") as f:
        for event in events:
            data = event.SerializeToString()
            header = struct.pack("Q", len(data))
            f.write(header)
            f.write(struct.pack("I", _masked_crc32c(header)))
            f.write(data)
            f.write(struct.pack("I", _masked_crc32c(data)))



class TBLog:
    def __init__(self, path: str):
        """Load a TensorBoard log directory."""
        self.path = path
        self.events = {}  # fname -> list of Event
        self.hparams = {}  # flat dict of hparam key -> value

        self._load()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self._apply_hparams()
        for fname, events in self.events.items():
            out_file = os.path.join(path, fname)
            _write_events(out_file, events)
        print(f"Saved → {path}")

    def __repr__(self):
        n_events = sum(len(v) for v in self.events.values())
        return (
            f"TBLog(path={self.path!r}, "
            f"files={list(self.events.keys())}, "
            f"events={n_events}, "
            f"hparams={self.hparams})"
        )

    def match_cfg(self):
        cfg = QuadrotorEnvConfig()
        cfg_dict = cfg.to_dict()
        for key in cfg_dict:
            if key not in self.hparams:
                self.hparams[key] = cfg_dict[key]
    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _load(self):
        for fname in sorted(os.listdir(self.path)):
            if not fname.startswith("events.out.tfevents"):
                continue
            fpath = os.path.join(self.path, fname)
            self.events[fname] = list(self._read_events(fpath))

        self.hparams = self._extract_hparams()

    @staticmethod
    def _read_events(path):
        with open(path, "rb") as f:
            while True:
                header = f.read(8)
                if len(header) < 8:
                    break
                data_len = struct.unpack("Q", header)[0]
                f.read(4)           # header crc
                data = f.read(data_len)
                f.read(4)           # data crc
                event = Event()
                event.ParseFromString(data)
                yield event

    def _extract_hparams(self):
        hparams = {}
        for events in self.events.values():
            for event in events:
                if not event.HasField("summary"):
                    continue
                for value in event.summary.value:
                    if value.metadata.plugin_data.plugin_name != "hparams":
                        continue
                    content = plugin_data_pb2.HParamsPluginData()
                    content.ParseFromString(value.metadata.plugin_data.content)
                    if content.HasField("session_start_info"):
                        for key, hp in content.session_start_info.hparams.items():
                            if hp.HasField("number_value"):
                                hparams[key] = hp.number_value
                            elif hp.HasField("string_value"):
                                hparams[key] = hp.string_value
                            elif hp.HasField("bool_value"):
                                hparams[key] = hp.bool_value
        return hparams

    def _extract_hparams(self):
        hparams = {}
        for events in self.events.values():
            for event in events:
                if not event.HasField("summary"):
                    continue
                for value in event.summary.value:
                    if value.metadata.plugin_data.plugin_name != "hparams":
                        continue
                    content = HParamsPluginData()
                    content.ParseFromString(value.metadata.plugin_data.content)
                    if content.HasField("session_start_info"):
                        for key, hp in content.session_start_info.hparams.items():
                            if hp.HasField("number_value"):
                                hparams[key] = hp.number_value
                            elif hp.HasField("string_value"):
                                hparams[key] = hp.string_value
                            elif hp.HasField("bool_value"):
                                hparams[key] = hp.bool_value
        return hparams

    def _make_hparam_value(self, val):
        """Convert a Python value to a protobuf Value."""
        from tensorboard.plugins.hparams.api_pb2 import HParamInfo
        from google.protobuf.struct_pb2 import Value
        hp = Value()
        if isinstance(val, bool):
            hp.bool_value = val
        elif isinstance(val, (int, float)):
            hp.number_value = float(val)
        else:
            hp.string_value = str(val)
        return hp

    def _apply_hparams(self):
        """Write self.hparams back into event protos, adding new keys as needed."""
        for events in self.events.values():
            for event in events:
                if not event.HasField("summary"):
                    continue
                for value in event.summary.value:
                    if value.metadata.plugin_data.plugin_name != "hparams":
                        continue
                    content = HParamsPluginData()
                    content.ParseFromString(value.metadata.plugin_data.content)

                    # --- patch experiment: register any new hparam keys in schema ---
                    if content.HasField("experiment"):
                        existing_keys = {h.name for h in content.experiment.hparam_infos}
                        for key, val in self.hparams.items():
                            if key not in existing_keys:
                                from tensorboard.plugins.hparams.api_pb2 import HParamInfo, DataType
                                if isinstance(val, bool):
                                    dtype = DataType.DATA_TYPE_BOOL
                                elif isinstance(val, (int, float)):
                                    dtype = DataType.DATA_TYPE_FLOAT64
                                else:
                                    dtype = DataType.DATA_TYPE_STRING
                                content.experiment.hparam_infos.append(
                                    HParamInfo(name=key, type=dtype)
                                )

                    # --- patch session_start_info: set all hparam values ---
                    if content.HasField("session_start_info"):
                        for key, val in self.hparams.items():
                            hp = content.session_start_info.hparams[key]  # auto-creates if missing
                            if isinstance(val, bool):
                                hp.bool_value = val
                            elif isinstance(val, (int, float)):
                                hp.number_value = float(val)
                            else:
                                hp.string_value = str(val)

                    value.metadata.plugin_data.content = content.SerializeToString()

# def main():
#     print(os.getcwd())
#     ROOT = "quad_experiment2/tb_test"
#     for dir in os.listdir(ROOT):
#         path = ROOT+"/"+dir
#         tblog = TBLog(path)
#         tblog.match_cfg()
#         tblog.save(path)
#     pass

def main():
    if len(sys.argv) < 2:
        print(f"current dir: {os.getcwd()}")
        print("Usage: python script.py <root_dir>")
        sys.exit(1)

    ROOT = sys.argv[1]

    if not os.path.isdir(ROOT):
        print(f"Error: '{ROOT}' is not a valid directory.")
        sys.exit(1)

    # Collect all valid TBLog directories
    runs = []
    for dir in sorted(os.listdir(ROOT)):
        path = os.path.join(ROOT, dir)
        if not os.path.isdir(path):
            continue
        has_events = any(f.startswith("events.out.tfevents") for f in os.listdir(path))
        if not has_events:
            continue
        runs.append(path)

    if not runs:
        print(f"No TensorBoard logs found in '{ROOT}'.")
        sys.exit(1)

    # Load all logs and preview changes
    logs = []
    print(f"\nFound {len(runs)} TensorBoard log(s) in '{ROOT}':\n")
    for path in runs:
        tblog = TBLog(path)
        tblog.match_cfg()
        before = TBLog(path).hparams  # reload original for comparison
        after = tblog.hparams

        new_keys = {k: v for k, v in after.items() if k not in before}
        changed_keys = {k: (before[k], v) for k, v in after.items() if k in before and before[k] != v}

        print(f"  {path}")
        if not new_keys and not changed_keys:
            print("    (no changes)")
        for k, v in new_keys.items():
            print(f"    + {k} = {v}  [new]")
        for k, (old, new) in changed_keys.items():
            print(f"    ~ {k}: {old} → {new}")
        print()

        logs.append(tblog)

    # Confirm
    answer = input("Apply changes and save all? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        sys.exit(0)

    for tblog in logs:
        tblog.save(tblog.path)

    print(f"\nDone. Updated {len(logs)} log(s).")

if __name__ == "__main__":
    main()