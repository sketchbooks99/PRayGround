#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <string_view>

namespace prayground {
    namespace util {
        template <typename Key, typename Value, size_t Size>
        class ConstexprMap {
        private:
            static_assert(std::is_invocable_v<std::less<>, Key, Key>, "Key must be comparable");
            using ValueT = std::pair<Key, Value>;
            using ContainerT = std::array<ValueT, Size>;

        public:
            template <size_t... Indices>
            constexpr ConstexprMap(const ValueT(&values)[Size], std::index_sequence<Indices...>) noexcept
                : m_values{ {{values[Indices].first, values[Indices].second}... } }
            {
                for (size_t i = 0; i < m_values.size() - 1; i++)
                {
                    for (size_t j = 0; j < m_values.size() - 1 - i; j++)
                    {
                        if (m_values[j + 1].first < m_values[j].first)
                        {
                            auto tmp = m_values[j];
                            m_values[j].first = m_values[j + 1].first;
                            m_values[j].second = m_values[j + 1].second;
                            m_values[j + 1].first = tmp.first;
                            m_values[j + 1].second = tmp.second;
                        }
                    }
                }
            }

            constexpr auto begin() const noexcept { return m_values.begin(); }
            constexpr auto end() const noexcept { return m_values.end(); }

            constexpr auto find(const Key& key) const noexcept -> typename ContainerT::const_iterator
            {
                auto left = begin();
                auto right = end();
                while (left < right)
                {
                    const auto mid = left + (right - left) / 2;
                    if (mid->first < key)
                        left = mid + 1;
                    else
                        right = mid;
                }

                if (left != end() && left->first == key)
                    return left;

                return m_values.end();
            }

            constexpr Value operator[](const Key& key) const noexcept
            {
                return find(key)->second;
            }
        private:
            ContainerT m_values;
        };

        template <typename Key, typename Value, size_t Size>
        inline constexpr auto makeConstexprMap(const std::pair<Key, Value>(&values)[Size])
        {
            ConstexprMap<Key, Value, Size> ret{values, std::make_index_sequence<Size>{}};
            return ret;
        }
    } // namespace util
} // namespace prayground