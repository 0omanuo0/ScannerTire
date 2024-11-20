# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "compute_dxData",
        ["compute_dxData.pyx"],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="compute_dxData",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",  # Establece el nivel de lenguaje a Python 3
            "boundscheck": False,    # Desactiva la verificación de límites
            "wraparound": False,     # Desactiva el wraparound para índices negativos
            "cdivision": True,       # Habilita la división C
        },
    ),
)