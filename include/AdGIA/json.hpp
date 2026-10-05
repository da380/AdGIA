/**
 * @file json.hpp
 * @brief A small JSON reader: the whole of the language, except that
 * \\u escapes beyond ASCII become '?' and numbers are read by strtod (which
 * also accepts some non-JSON forms, such as hexadecimal and inf/nan).
 *
 * Shared by MeshManifest and the benchmark drivers, which read case files
 * written by Python. Values are a plain tree (Json); errors abort with the
 * file name and the character position.
 */

#pragma once

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "mfem.hpp"

namespace AdGIA {

/** @brief A JSON value. Object members keep their order. */
struct Json {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string string;
  std::vector<Json> array;
  std::vector<std::pair<std::string, Json>> object;

  const Json* Find(const std::string& key) const {
    for (const auto& [k, v] : object) {
      if (k == key) {
        return &v;
      }
    }
    return nullptr;
  }
  bool IsNull() const { return type == Type::Null; }
  bool IsInteger() const {
    return type == Type::Number && number == std::floor(number);
  }
};

/** @brief Parser of a JSON document. */
class JsonReader {
 public:
  /** @param text The document (must outlive the reader).
   * @param file Its name, for the error messages. */
  JsonReader(const std::string& text, std::string file)
      : s_(text), file_(std::move(file)) {}

  Json Read() {
    Json v = Value();
    Space();
    if (i_ != s_.size()) {
      Fail("text after the end of the document");
    }
    return v;
  }

 private:
  [[noreturn]] void Fail(const std::string& what) const {
    MFEM_ABORT(file_ << " is not valid JSON: " << what
                                << " at character " << i_ << ".");
  }

  void Space() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' ||
                              s_[i_] == '\t' || s_[i_] == '\r')) {
      i_++;
    }
  }

  char Peek() {
    Space();
    if (i_ >= s_.size()) {
      Fail("unexpected end");
    }
    return s_[i_];
  }

  void Expect(char c) {
    if (Peek() != c) {
      Fail(std::string("expected '") + c + "'");
    }
    i_++;
  }

  bool Literal(const char* word) {
    const std::string w(word);
    if (s_.compare(i_, w.size(), w) == 0) {
      i_ += w.size();
      return true;
    }
    return false;
  }

  std::string String() {
    Expect('"');
    std::string out;
    while (true) {
      if (i_ >= s_.size()) {
        Fail("unterminated string");
      }
      const char c = s_[i_++];
      if (c == '"') {
        return out;
      }
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i_ >= s_.size()) {
        Fail("unterminated escape");
      }
      const char e = s_[i_++];
      switch (e) {
        case '"':
        case '\\':
        case '/':
          out += e;
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u': {
          if (i_ + 4 > s_.size()) {
            Fail("short \\u escape");
          }
          const unsigned long code =
              std::strtoul(s_.substr(i_, 4).c_str(), nullptr, 16);
          i_ += 4;
          out += code < 128 ? static_cast<char>(code) : '?';
          break;
        }
        default:
          Fail("unknown escape");
      }
    }
  }

  Json Value() {
    Json v;
    const char c = Peek();
    if (c == '{') {
      v.type = Json::Type::Object;
      i_++;
      if (Peek() == '}') {
        i_++;
        return v;
      }
      while (true) {
        Space();
        std::string key = String();
        Expect(':');
        v.object.emplace_back(std::move(key), Value());
        if (Peek() == ',') {
          i_++;
          continue;
        }
        Expect('}');
        return v;
      }
    }
    if (c == '[') {
      v.type = Json::Type::Array;
      i_++;
      if (Peek() == ']') {
        i_++;
        return v;
      }
      while (true) {
        v.array.push_back(Value());
        if (Peek() == ',') {
          i_++;
          continue;
        }
        Expect(']');
        return v;
      }
    }
    if (c == '"') {
      v.type = Json::Type::String;
      v.string = String();
      return v;
    }
    if (Literal("true")) {
      v.type = Json::Type::Bool;
      v.boolean = true;
      return v;
    }
    if (Literal("false")) {
      v.type = Json::Type::Bool;
      return v;
    }
    if (Literal("null")) {
      return v;
    }
    const char* begin = s_.c_str() + i_;
    char* end = nullptr;
    v.number = std::strtod(begin, &end);
    if (end == begin) {
      Fail("unexpected character");
    }
    v.type = Json::Type::Number;
    i_ += end - begin;
    return v;
  }

  const std::string& s_;
  std::string file_;
  std::size_t i_ = 0;
};

/** @brief Read and parse the JSON file @p path (aborts on failure). */
inline Json ReadJsonFile(const std::string& path) {
  std::ifstream in(path);
  MFEM_VERIFY(in.good(), "Cannot open " << path << ".");
  std::stringstream buffer;
  buffer << in.rdbuf();
  const std::string text = buffer.str();
  return JsonReader(text, path).Read();
}

}  // namespace AdGIA
