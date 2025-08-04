# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module provides a metaclass and a decorator to prevent overriding of methods and properties
in subclasses. The main use-case is to fix the concrete implementations of the non-abstract methods
and properties in the base classes, so that they cannot be overridden in the subclasses.
"""
from abc import ABCMeta
from typing import Callable


class NoOverrideMeta(ABCMeta):
    """Metaclass to prevent overriding of methods in subclasses."""

    def __new__(cls, name: str, bases: tuple, dct: dict) -> type:
        """
        Create a new class instance and ensure that methods/properties marked with _no_override
        in the base classes are not overridden in the subclass.

        Args:
            cls: The metaclass.
            name: The name of the new class.
            bases: A tuple of base classes used for creating the new class.
            dct: A dictionary of class methods and attributes.

        Returns:
            The newly created class instance.

        Raises:
            TypeError: If a method marked with _no_override in a base class is overridden
                       in the subclass.
        """
        # Check for overridden methods
        implemented_methods = [attr_name for (attr_name, attr_value) in dct.items()]
        for base in bases:
            for attr_name, attr_value in base.__dict__.items():
                # In case attribute value is a property we should check for _no_override in the
                # getter function of that property. For other cases we can check for _no_override in
                # attribute itself.
                target = attr_value.fget if isinstance(attr_value, property) else attr_value
                if getattr(target, "_no_override", False) and attr_name in implemented_methods:
                    raise TypeError(
                        f"Method '{attr_name}' in {base.__name__} cannot be overridden in {name}"
                    )
        return super().__new__(cls, name, bases, dct)


def no_override(method_or_property: Callable) -> Callable:
    """
    Decorator to mark a method/property as non-overridable.

    Notes: in case of property, the getter function is marked as non-overridable.

    Args:
        method_or_property: the method or the property which should be decorated.

    Returns:
        the provided method or property is returned, only this time it has a property _no_override
          which it is set to True.
    """
    if isinstance(method_or_property, property):
        method_or_property.fget._no_override = True
        return method_or_property  # Return the property object itself
    else:
        method_or_property._no_override = True  # type: ignore[attr-defined]
        return method_or_property
