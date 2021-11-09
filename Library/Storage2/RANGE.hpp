#pragma once

#include <cstddef>
#include <vector>

namespace JGSL {
namespace storage {
  struct Range {
    std::size_t start, amount;

    Range(std::size_t start, std::size_t amount) : start(start), amount(amount) {}

    Range() : Range(0, 0) {}

    inline bool intersects(const Range &other) const {
      return end() > other.start && start < other.end();
    }

    inline bool operator<=(const Range &other) const {
      return end() <= other.end();
    }

    inline std::size_t begin() const {
      return start;
    }

    inline std::size_t end() const {
      return start + amount;
    }

    inline std::size_t size() const {
      return amount;
    }

    inline bool contains(std::size_t i) const {
      return i >= begin() && i < end();
    }

    Range intersect(const Range &other) const {
      std::size_t res_start = std::max(start, other.start);
      std::size_t res_end = std::min(end(), other.end());
      if (res_start > res_end) {
        return Range();
      } else {
        return Range(res_start, res_end - res_start);
      }
    }

    std::string to_string() const {
      return "[" + std::to_string(start) + ", " + std::to_string(end()) + ")";
    }
  };

  struct Ranges {
    bool is_infinity;
    std::vector<Range> ranges;

    Ranges() : is_infinity(false) {}

    Ranges(bool is_infinity) : is_infinity(is_infinity) {}

    inline Range &operator[](std::size_t i) {
      return ranges[i];
    }

    inline const Range &operator[](std::size_t i) const {
      return ranges[i];
    }

    std::size_t add(const Range &range) {
      ranges.push_back(range);
      if (ranges.size() > 1) {
        int curr = ranges.size() - 1;
        while (--curr >= 0 && range <= ranges[curr]) {
          std::swap(ranges[curr], ranges[curr + 1]);
        }
        return curr + 1;
      } else {
        return 0;
      }
    }

    inline std::size_t add(std::size_t start, std::size_t amount) {
      return add(Range(start, amount));
    }

    inline auto begin() const {
      return ranges.begin();
    }

    inline auto end() const {
      return ranges.end();
    }

    bool contains(std::size_t i) const {
      for (auto range : ranges) {
        if (range.contains(i))
          return true;
      }
      return false;
    }

    Ranges intersect(const Ranges &other) const {

      // Make infinity a special case
      if (is_infinity) {
        return other;
      }
      if (other.is_infinity) {
        return *this;
      }

      // Calculate result afterwards
      Ranges result;
      std::size_t i = 0, j = 0;
      while (i < ranges.size() && j < other.ranges.size()) {
        const Range &curr_i = ranges[i];
        const Range &curr_j = other.ranges[j];

        // Check if there's intersection. If there is, add it to the result
        if (curr_i.intersects(curr_j)) {
          result.add(curr_i.intersect(curr_j));
        }

        // Move forward
        if (curr_i <= curr_j) {
          i++;
        } else {
          j++;
        }
      }
      return result;
    }

    std::string to_string() const {
      if (is_infinity) {
        return "inf";
      }

      std::string result = "[";
      for (std::size_t i = 0; i < ranges.size(); i++) {
        result += ranges[i].to_string();
        if (i != ranges.size() - 1) {
          result += ", ";
        }
      }
      result += "]";
      return result;
    }
  };

  struct RangesMap {
    Ranges globals;
    std::vector<std::size_t> locals;

    void add(std::size_t local, const Range &global) {
      std::size_t index = globals.add(global);
      std::size_t prev = local;
      for (std::size_t i = index; i < locals.size(); i++) {
        std::swap(prev, locals[i]);
      }
      locals.push_back(prev);
    }

    inline void add(std::size_t local, std::size_t global, std::size_t amount) {
      add(local, Range(global, amount));
    }

    std::size_t to_local(std::size_t global) const {
      for (std::size_t i = 0; i < locals.size(); i++) {
        if (globals.ranges[i].contains(global)) {
          return locals[i] + global - globals.ranges[i].start;
        }
      }
      return (std::size_t)-1;
    }

    std::string to_string() const {
      std::string result = "[";
      for (std::size_t i = 0; i < locals.size(); i++) {
        result += "(" + std::to_string(locals[i]) + ", " + globals.ranges[i].to_string() + ")";
        if (i != locals.size() - 1) {
          result += ", ";
        }
      }
      result += "]";
      return result;
    }
  };
} // namespace storage
}