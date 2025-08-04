import shutil

if not shutil.which("julia"):
    from jill.install import install_julia

    install_julia(version="1.11.3", confirm=True)

# Import diffeqpy.de to install required Julia packages
from diffeqpy import de  # noqa: F401
