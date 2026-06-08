from pathlib import Path

from scene import Scene


class Dataset:
    def __init__(self, root):
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(self.root)

        self._objects = self._scan()

    @property
    def class_names(self):
        return sorted(self._objects.keys())

    @property
    def scenes(self):
        return [p for blends in self._objects.values() for p in blends]

    def _scan(self):
        objects = {}

        for class_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            blends = sorted(
                p for p in class_dir.iterdir()
                if p.suffix == '.blend')

            if blends:
                objects[class_dir.name] = blends

        return objects

    def get_scene(self, identifier, **scene_kwargs):
        try:
            class_name = identifier.split('_')[0]
        except ValueError:
            raise ValueError(
                f"Invalid identifier '{identifier}'. "
                "Expected format '<class>_<id>'"
            )
        print(f'Getting scene: {identifier} (class: {class_name})')
        path_class = self.root / class_name
        assert path_class.exists(), f'Class not found: {class_name}'

        path_blend = path_class / f'{identifier}.blend'
        assert path_blend.exists(), f'Scene not found: {identifier}'

        return Scene(path_blend=path_blend, **scene_kwargs)

    def objects_in_class(self, class_name):
        try:
            return [p.stem for p in self._objects[class_name]]
        except KeyError:
            raise KeyError(f'Unknown class: {class_name}')

    def __len__(self):
        return sum(len(v) for v in self._objects.values())

    def __str__(self):
        return (
            'Dataset\n'
            f'  root:    {self.root}\n'
            f'  classes: {len(self.class_names)}\n'
            f'  scenes:  {len(self.scenes)}'
        )
