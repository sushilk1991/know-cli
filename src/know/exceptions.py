"""Custom exceptions for know-cli."""


class KnowError(Exception):
    """Base exception for know-cli."""
    pass


class ConfigError(KnowError):
    """Configuration-related errors."""
    pass


class ScanError(KnowError):
    """Code scanning errors."""
    pass


class ParseError(ScanError):
    """File parsing errors."""
    
    def __init__(self, message: str, path: str, line: int = 0):
        super().__init__(message)
        self.path = path
        self.line = line
    
    def __str__(self) -> str:
        if self.line:
            return f"{self.path}:{self.line}: {self.args[0]}"
        return f"{self.path}: {self.args[0]}"


class IndexError(KnowError):
    """Index/database errors."""
    pass


class AIError(KnowError):
    """AI/API-related errors."""
    pass


class WatchError(KnowError):
    """File watching errors."""
    pass


class ValidationError(KnowError):
    """Data validation errors."""
    pass
