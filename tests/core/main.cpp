#include <prayground/core/util.h>
#include <optional>

using namespace std;
using namespace prayground;

int main()
{
    optional<int> a;
    a = nullopt;

    ASSERT(a, "This is nullopt");
    THROW("Fail");

    return 0;
}