#include "file_util.h"
#include <optional>
#include <array>
#include <prayground/core/util.h>

namespace prayground {

    namespace fs = std::filesystem;

    namespace {
        // Application directory
        fs::path app_dir = fs::path("");

    } // nonamed namespace

    // -------------------------------------------------------------------------------
    std::optional<fs::path> pgFindDataPath( const fs::path& relative_path )
    {
        std::array<std::string, 4> parent_dirs = 
        {
            pgAppDir().string(), 
            pgPathJoin(pgAppDir(), "data").string(), 
            "",
            pgRootDir().string(),
        };

        for (auto &parent : parent_dirs)
        {
            auto filepath = pgPathJoin(parent, relative_path);
            if ( fs::exists(filepath) )
                return filepath;
        }
        return std::nullopt;
    }

    // -------------------------------------------------------------------------------
    std::string pgGetLowerString(std::string str)
    {
        // Replace all letters to lower character
        std::transform(str.begin(), str.end(), str.begin(), 
            [](char c) { return std::tolower(c); });
        return str;
    }

    // -------------------------------------------------------------------------------
    fs::path pgRootDir() {
        return fs::path(PRAYGROUND_DIR);
    }

    void pgSetAppDir(const fs::path& dir)
    {
        app_dir = dir;
    }

    fs::path pgAppDir()
    {
        return app_dir;
    }

    // -------------------------------------------------------------------------------
    std::string pgGetExtension( const fs::path& filepath )
    {
        return filepath.has_extension() ? filepath.extension().string() : "";
    }

    std::string pgGetStem(const fs::path& filepath, bool is_dir)
    {
        std::string dirpath = filepath.has_parent_path() ? filepath.parent_path().string() : "";
        std::string stem = filepath.stem().string();
        return is_dir ? dirpath + "/" + stem : stem;
    }

    std::filesystem::path pgGetDir(const fs::path& filepath)
    {
        return filepath.has_parent_path() ? filepath.parent_path() : "";
    }

    // -------------------------------------------------------------------------------
    void pgCreateDir( const fs::path& abs_path )
    {
        // Check if the directory is existed.
        if (fs::exists(abs_path)) {
            pgLogWarn("The directory '", abs_path, "' is already existed.");
            return;
        }
        // Create new directory with path specified.
        bool result = fs::create_directory(abs_path);
        ASSERT(result, "Failed to create directory '" + abs_path.string() + "'.");
    }

    // -------------------------------------------------------------------------------
    void pgCreateDirs( const fs::path& abs_path )
    {
        // Check if the directory is existed.
        if (fs::exists(abs_path)) {
            pgLogWarn("The directory '", abs_path, "' is already existed.");
            return;
        }
        bool result = fs::create_directories( abs_path );
        ASSERT(result, "Failed to create directories '" + abs_path.string() + "'.");
    }

    // -------------------------------------------------------------------------------
    std::string pgGetTextFromFile(const fs::path& relative_path)
    {
        std::optional<fs::path> filepath = pgFindDataPath(relative_path);
        ASSERT(filepath, "A text file with the path '" + relative_path.string() + "' is not found.");

        std::ifstream file_stream; 
        try
        {
            file_stream.open(filepath.value());
            std::stringstream content_stream;
            content_stream << file_stream.rdbuf();
            file_stream.close();
            return content_stream.str();
        }
        catch(const std::istream::failure& e)
        {
            pgLogFatal("Failed to load text file due to '" + std::string(e.what()) + "'.");
            return "";
        }
    }

} // namespace prayground