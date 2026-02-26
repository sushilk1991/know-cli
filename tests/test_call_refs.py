"""Tests for cross-language call reference extraction.

Covers TreeSitterParser and regex fallback for all supported languages.
"""

import tempfile
import shutil
from pathlib import Path

import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from know.parsers import (
    ParserFactory,
    PythonParser,
    TypeScriptParser,
    GoParser,
    TreeSitterParser,
    TypeScriptRegexParser,
    GoRegexParser,
    RustRegexParser,
    JavaRegexParser,
    RubyRegexParser,
    CRegexParser,
    SwiftRegexParser,
    _get_ts_parser,
)


@pytest.fixture
def temp_dir():
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _ref_names(refs):
    """Extract set of ref_name from call ref list."""
    return {r["ref_name"] for r in refs}


def _refs_in_chunk(refs, chunk_name):
    """Get ref_names for a specific containing_chunk."""
    return {r["ref_name"] for r in refs if r["containing_chunk"] == chunk_name}


# ---------------------------------------------------------------------------
# Python (AST-based)
# ---------------------------------------------------------------------------

class TestPythonCallRefs:
    def test_simple_calls(self, temp_dir):
        parser = PythonParser()
        code = temp_dir / "example.py"
        code.write_text("""\
def greet(name):
    message = format_name(name)
    send_email(message)
    return message

def format_name(n):
    return n.upper()
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "format_name" in names
        assert "send_email" in names

    def test_method_calls(self, temp_dir):
        parser = PythonParser()
        code = temp_dir / "example.py"
        code.write_text("""\
class Service:
    def process(self):
        result = self.validate()
        self.save(result)

    def validate(self):
        return True

    def save(self, data):
        pass
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "validate" in names
        assert "save" in names

    def test_containing_chunk(self, temp_dir):
        parser = PythonParser()
        code = temp_dir / "example.py"
        code.write_text("""\
def caller():
    target()

def target():
    pass
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        in_caller = _refs_in_chunk(refs, "caller")
        assert "target" in in_caller

    def test_import_alias_resolution(self, temp_dir):
        parser = PythonParser()
        code = temp_dir / "example.py"
        code.write_text("""\
from os.path import join as path_join

def build_path():
    return path_join("a", "b")
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        # Should resolve alias to original name
        assert "join" in names


# ---------------------------------------------------------------------------
# TypeScript / JavaScript
# ---------------------------------------------------------------------------

class TestTypeScriptCallRefs:
    def test_regex_call_refs(self, temp_dir):
        parser = TypeScriptRegexParser()
        code = temp_dir / "app.ts"
        code.write_text("""\
function processData(input: string) {
    const result = validate(input);
    return transform(result);
}

function validate(s: string) {
    return s.trim();
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "validate" in names
        assert "transform" in names

    def test_regex_jsx_component_refs(self, temp_dir):
        parser = TypeScriptRegexParser()
        parser.language = "tsx"
        code = temp_dir / "page.tsx"
        code.write_text("""\
export function Page() {
    return (
        <div>
            <Header />
            <Sidebar items={items} />
            <Footer />
        </div>
    );
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "Header" in names
        assert "Sidebar" in names
        assert "Footer" in names

    def test_regex_skips_keywords(self, temp_dir):
        parser = TypeScriptRegexParser()
        code = temp_dir / "logic.ts"
        code.write_text("""\
function check(x: number) {
    if (x > 0) {
        return doWork(x);
    }
    while (running()) {
        wait();
    }
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "doWork" in names
        assert "if" not in names
        assert "while" not in names

    def test_treesitter_call_refs(self, temp_dir):
        """Test tree-sitter based extraction when available."""
        parser, _ = _get_ts_parser("typescript")
        if parser is None:
            pytest.skip("tree-sitter-typescript not installed")
        ts_parser = TreeSitterParser("typescript")
        code = temp_dir / "service.ts"
        code.write_text("""\
function fetchData(url: string) {
    const response = httpGet(url);
    return parseJSON(response);
}
""")
        module = ts_parser.parse(code, temp_dir)
        refs = ts_parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "httpGet" in names
        assert "parseJSON" in names


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

class TestGoCallRefs:
    def test_regex_call_refs(self, temp_dir):
        parser = GoRegexParser()
        code = temp_dir / "main.go"
        code.write_text("""\
package main

func handler(w http.ResponseWriter, r *http.Request) {
    data := fetchData(r)
    writeResponse(w, data)
}

func fetchData(r *http.Request) string {
    return parseBody(r)
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "fetchData" in names
        assert "writeResponse" in names
        assert "parseBody" in names

    def test_regex_skips_go_builtins(self, temp_dir):
        parser = GoRegexParser()
        code = temp_dir / "util.go"
        code.write_text("""\
package main

func process() {
    s := make([]int, 10)
    n := len(s)
    s = append(s, 42)
    doWork(n)
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "doWork" in names
        assert "make" not in names
        assert "len" not in names
        assert "append" not in names

    def test_treesitter_call_refs(self, temp_dir):
        parser, _ = _get_ts_parser("go")
        if parser is None:
            pytest.skip("tree-sitter-go not installed")
        ts_parser = TreeSitterParser("go")
        code = temp_dir / "main.go"
        code.write_text("""\
package main

import "fmt"

func main() {
    result := compute(42)
    fmt.Println(result)
}

func compute(n int) int {
    return helper(n) + 1
}

func helper(n int) int {
    return n * 2
}
""")
        module = ts_parser.parse(code, temp_dir)
        refs = ts_parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "compute" in names
        assert "helper" in names
        # fmt.Println -> should extract "Println"
        assert "Println" in names

    def test_go_parser_delegates(self, temp_dir):
        """GoParser should delegate to tree-sitter or use regex fallback."""
        parser = GoParser(use_treesitter=False)
        code = temp_dir / "main.go"
        code.write_text("""\
package main

func doStuff() {
    callA()
    callB()
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "callA" in names
        assert "callB" in names


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------

class TestRustCallRefs:
    def test_regex_call_refs(self, temp_dir):
        parser = RustRegexParser()
        code = temp_dir / "lib.rs"
        code.write_text("""\
fn process(data: &str) -> Result<String, Error> {
    let parsed = parse_input(data)?;
    let result = transform(parsed);
    Ok(result)
}

fn parse_input(s: &str) -> Result<Data, Error> {
    serde_json::from_str(s)
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "parse_input" in names
        assert "transform" in names

    def test_regex_skips_rust_keywords(self, temp_dir):
        parser = RustRegexParser()
        code = temp_dir / "lib.rs"
        code.write_text("""\
fn check(x: i32) {
    if x > 0 {
        do_work(x);
    }
    match get_value() {
        Some(v) => handle(v),
        None => panic!("nope"),
    }
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "do_work" in names
        assert "get_value" in names
        assert "handle" in names
        assert "if" not in names
        assert "match" not in names

    def test_treesitter_method_call(self, temp_dir):
        parser, _ = _get_ts_parser("rust")
        if parser is None:
            pytest.skip("tree-sitter-rust not installed")
        ts_parser = TreeSitterParser("rust")
        code = temp_dir / "lib.rs"
        code.write_text("""\
fn process(items: Vec<Item>) -> Vec<String> {
    items.iter()
        .filter(|i| is_valid(i))
        .map(|i| format_item(i))
        .collect()
}

fn is_valid(item: &Item) -> bool {
    validate(item)
}
""")
        module = ts_parser.parse(code, temp_dir)
        refs = ts_parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "is_valid" in names
        assert "format_item" in names
        # Chained method calls via field_expression inside call_expression
        assert "iter" in names
        assert "collect" in names


# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------

class TestJavaCallRefs:
    def test_regex_call_refs(self, temp_dir):
        parser = JavaRegexParser()
        code = temp_dir / "Service.java"
        code.write_text("""\
public class Service {
    public void process(String input) {
        String validated = validate(input);
        save(validated);
    }

    private String validate(String s) {
        return sanitize(s);
    }
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "validate" in names
        assert "save" in names
        assert "sanitize" in names

    def test_regex_skips_java_keywords(self, temp_dir):
        parser = JavaRegexParser()
        code = temp_dir / "Main.java"
        code.write_text("""\
public class Main {
    public void run() {
        if (check()) {
            for (int i = 0; i < 10; i++) {
                doWork(i);
            }
        }
    }
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "doWork" in names
        assert "check" in names
        assert "if" not in names
        assert "for" not in names

    def test_treesitter_call_refs(self, temp_dir):
        parser, _ = _get_ts_parser("java")
        if parser is None:
            pytest.skip("tree-sitter-java not installed")
        ts_parser = TreeSitterParser("java")
        code = temp_dir / "Service.java"
        code.write_text("""\
public class Service {
    public void handle(Request req) {
        Response resp = buildResponse(req);
        sendResponse(resp);
    }

    private Response buildResponse(Request req) {
        return new Response(req.getData());
    }
}
""")
        module = ts_parser.parse(code, temp_dir)
        refs = ts_parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "buildResponse" in names
        assert "sendResponse" in names


# ---------------------------------------------------------------------------
# Ruby
# ---------------------------------------------------------------------------

class TestRubyCallRefs:
    def test_regex_call_refs(self, temp_dir):
        parser = RubyRegexParser()
        code = temp_dir / "service.rb"
        code.write_text("""\
class Service
  def process(input)
    validated = validate(input)
    save(validated)
  end

  def validate(s)
    sanitize(s)
  end
end
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "validate" in names
        assert "save" in names
        assert "sanitize" in names

    def test_regex_skips_ruby_keywords(self, temp_dir):
        parser = RubyRegexParser()
        code = temp_dir / "app.rb"
        code.write_text("""\
def check(x)
  if x > 0
    do_work(x)
  end
  puts "done"
end
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "do_work" in names
        assert "if" not in names
        assert "puts" not in names

    def test_treesitter_call_refs(self, temp_dir):
        parser, _ = _get_ts_parser("ruby")
        if parser is None:
            pytest.skip("tree-sitter-ruby not installed")
        ts_parser = TreeSitterParser("ruby")
        code = temp_dir / "service.rb"
        code.write_text("""\
class Service
  def process(input)
    validated = validate(input)
    transform(validated)
  end

  def validate(s)
    sanitize(s)
  end
end
""")
        module = ts_parser.parse(code, temp_dir)
        refs = ts_parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "validate" in names
        assert "transform" in names


# ---------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------

class TestCCallRefs:
    def test_regex_call_refs(self, temp_dir):
        parser = CRegexParser()
        code = temp_dir / "main.c"
        code.write_text("""\
#include <stdio.h>

int process(int x) {
    int y = compute(x);
    return transform(y);
}

int compute(int n) {
    return helper(n) + 1;
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "compute" in names
        assert "transform" in names
        assert "helper" in names

    def test_regex_skips_c_keywords(self, temp_dir):
        parser = CRegexParser()
        code = temp_dir / "util.c"
        code.write_text("""\
void check(int x) {
    if (x > 0) {
        while (running()) {
            do_work(x);
        }
    }
    int size = sizeof(x);
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "do_work" in names
        assert "running" in names
        assert "if" not in names
        assert "while" not in names
        assert "sizeof" not in names


# ---------------------------------------------------------------------------
# Swift
# ---------------------------------------------------------------------------

class TestSwiftCallRefs:
    def test_regex_call_refs(self, temp_dir):
        parser = SwiftRegexParser()
        code = temp_dir / "app.swift"
        code.write_text("""\
import Foundation

func process(input: String) -> String {
    let validated = validate(input)
    return transform(validated)
}

func validate(_ s: String) -> String {
    return sanitize(s)
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "validate" in names
        assert "transform" in names
        assert "sanitize" in names

    def test_regex_skips_swift_keywords(self, temp_dir):
        parser = SwiftRegexParser()
        code = temp_dir / "logic.swift"
        code.write_text("""\
func check(x: Int) {
    if x > 0 {
        doWork(x)
    }
    guard isValid(x) else { return }
    for item in getItems() {
        handle(item)
    }
}
""")
        module = parser.parse(code, temp_dir)
        refs = parser.extract_call_refs(code.read_text(), module)
        names = _ref_names(refs)
        assert "doWork" in names
        assert "isValid" in names
        assert "getItems" in names
        assert "handle" in names
        assert "if" not in names
        assert "guard" not in names


# ---------------------------------------------------------------------------
# ParserFactory integration
# ---------------------------------------------------------------------------

class TestParserFactoryCallRefs:
    """Verify parsers returned by factory have extract_call_refs."""

    @pytest.mark.parametrize("lang", [
        "python", "typescript", "tsx", "javascript", "jsx",
        "go", "rust", "java", "ruby", "c", "cpp", "swift",
    ])
    def test_factory_parser_has_extract_call_refs(self, lang):
        parser = ParserFactory.get_parser(lang, use_treesitter=False)
        if parser is None:
            pytest.skip(f"No parser for {lang}")
        assert hasattr(parser, "extract_call_refs"), (
            f"Parser for {lang} ({type(parser).__name__}) missing extract_call_refs"
        )

    @pytest.mark.parametrize("lang", [
        "python", "typescript", "tsx", "javascript", "jsx",
        "go", "rust", "java", "ruby", "c", "cpp",
    ])
    def test_factory_treesitter_parser_has_extract_call_refs(self, lang):
        parser = ParserFactory.get_parser(lang, use_treesitter=True)
        if parser is None:
            pytest.skip(f"No parser for {lang}")
        assert hasattr(parser, "extract_call_refs"), (
            f"Tree-sitter parser for {lang} ({type(parser).__name__}) missing extract_call_refs"
        )
