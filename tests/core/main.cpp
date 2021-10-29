#include <prayground/core/util.h>
#include <prayground/core/attribute.h>
#include <optional>

using namespace std;
using namespace prayground;

unique_ptr<string[]> make_string(const std::string& str)
{
    auto ptr = make_unique<string[]>(str.size());
    memcpy(ptr.get(), &str, str.size());
    return ptr;
}

void constructAttributes(vector<Attributes>& attribs)
{
    Attributes attrib;
    int n_attribs = static_cast<int>(attribs.size());
    attrib.addString(
        "str" + to_string(n_attribs), 
        make_string("str" + to_string(n_attribs)), 1);
    attribs.emplace_back(attrib);
}

class AttribContainer {
public:
    AttribContainer(){}
    void setup()
    {
        for (int i = 0; i < 20; i++)
            constructAttributes(attribs);
    }

    void run()
    {
        for (int i = 0; const auto& a : attribs)
        {
            std::string s = a.findOneString("str" + to_string(i), "");
            if (!s.empty())
                cout << s << endl;
            i++;
        }
    }
private:
    vector<Attributes> attribs;
};

int main()
{
    auto ac = make_shared<AttribContainer>();
    ac->setup();
    ac->run();

    return 0;
}