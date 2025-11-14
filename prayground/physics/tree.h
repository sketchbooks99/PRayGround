#pragma once

#include <prayground/math/vec.h>
#include <prayground/math/quat.h>
#include <prayground/math/bezier.h>
#include <prayground/core/onb.h>
#include <prayground/math/matrix.h>
#include <prayground/shape/trianglemesh.h>

#ifndef __CUDACC__
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <array>
#include <sstream>
#endif

namespace prayground {

#ifndef __CUDACC__
    enum class BranchMode {
        alt_opp = 1,
        whorled = 2,
        fan = 3
    };

    class Stem {
    public:
        // メンバ変数  
        int depth;                    // Branch depth（0=Stem、1=Main branch、2=Sub branch、3=Small branch）  
        std::shared_ptr<BezierSpline> curve;           // Bezier curve (owned by Stem)
        Stem* parent;                 // Pointer to parent branch
        float offset;                 // Starting position on parent branch 
        float radius_limit;           // Limit of radius 

        std::vector<std::shared_ptr<Stem>> children;  // List of branch children
        float length;                 // Length of branch
        float radius;                 // Radius of branch 
        float length_child_max;       // Maximum coefficient of children branch length

        Stem(int depth, Stem* parent = nullptr,
            float offset = 0.0f, float radius_limit = -1.0f)
            : depth(depth)
            , curve(std::make_shared<BezierSpline>())
            , parent(parent)
            , offset(offset)
            , radius_limit(radius_limit)
            , length(0.0f)
            , radius(0.0f)
            , length_child_max(0.0f)
        {

        }

        Stem(
            int depth, 
            std::shared_ptr<BezierSpline> curve,
            Stem* parent = nullptr,
            float offset = 0.0f, 
            float radius_limit = -1.0f, 
            float length = 0.0f,  
            float radius = 0.0f, 
            float length_child_max = 0.0f
        ) 
            : depth(depth)
            , curve(curve)
            , parent(parent)
            , offset(offset)
            , radius_limit(radius_limit)
            , length(length)
            , radius(radius)
            , length_child_max(length_child_max)
        { }
            
        // Copy constructor
        Stem(const Stem& other)
            : depth(other.depth)
            , curve(other.curve)
            , parent(other.parent)
            , offset(other.offset)
            , radius_limit(other.radius_limit)
            , children()
            , length(other.length)
            , radius(other.radius)
            , length_child_max(other.length_child_max)
        {
        }

        ~Stem() {}

        Stem copy() const {
            return Stem(depth, curve, parent, offset, radius_limit, length, radius, length_child_max);
        }

        std::string toString() const {
            std::ostringstream oss;
            oss << length << " " << offset << " " << radius;
            return oss.str();
        }
    };

    struct MeshData {
        std::vector<Vec3f> vertices;
        std::vector<std::vector<Face>> faces; // Multiple face groups
        std::vector<Vec2f> texcoords;
        std::vector<Vec3f> normals;
    };

    class Leaf {
    public:
        Leaf(const Vec3f& pos, const Vec3f& dir, const Vec3f& r, float radius = 0.0f) 
            : m_pos(pos), m_dir(dir), m_right(r), m_radius(radius) {}

        // Create quad for leaf
        static MeshData getShape(float g_scale, float scale, float scale_x);
        
        // Apply transformation (translate, rotate) to basic leaf shape
        MeshData getMesh(float bend, const MeshData& mesh, int index) const;

        std::pair<Quatf, Quatf> calcBendTransform(float bend) const;

        const Vec3f& position() const { return m_pos; }
        void setPosition(const Vec3f& pos) { m_pos = pos; }
        const Vec3f& direction() const { return m_dir; }
        void setDirection(const Vec3f& dir) { m_dir = dir; }
        const Vec3f& right() const { return m_right; }
        void setRight(const Vec3f& right) { m_right = right; }
        float radius() const { return m_radius; }
        void setRadius(float radius) { m_radius = radius; }
    private:
        Vec3f m_pos;
        Vec3f m_dir;
        Vec3f m_right;
        float m_radius;
    };

    class CHTurtle {
    public:
        CHTurtle();
        CHTurtle(const Vec3f& pos, const Vec3f& dir, const Vec3f& right, float width);
        CHTurtle(const CHTurtle& other);

        void move(float distance);
        void turnRight(float angle);
        void turnLeft(float angle);
        void pitchUp(float angle);
        void pitchDown(float angle);
        void rollRight(float angle);
        void rollLeft(float angle);

        const Vec3f& position() const { return m_pos; }
        void setPosition(const Vec3f& p) { m_pos = p; }

        const Vec3f& direction() const { return m_dir; }
        void setDirection(const Vec3f& d) { m_dir = d; }

        const Vec3f& right() const { return m_right; }
        void setRight(const Vec3f& r) { m_right = r; }

        float width() const { return m_width; }
        void setWidth(float w) { m_width = w; }
    private:
        Vec3f m_pos;
        Vec3f m_dir;
        Vec3f m_right;
        float m_width;
    };

    struct TreeParam {
        // Basic tree properties
        int shape = 8;
        float g_scale = 13.0f;
        float g_scale_v = 3.0f;
        float ratio = 0.015f;
        float ratio_power = 1.2f;
        float flare = 0.6f;
        int base_splits = 0;
        
        // Arrays for each level [0-3]

        int levels = 3;
        std::array<float, 4> base_size = {0.3f, 0.02f, 0.02f, 0.02f};
        std::array<float, 4> down_angle = {0.0f, 60.0f, 45.0f, 45.0f};
        std::array<float, 4> down_angle_v = {0.0f, -50.0f, 10.0f, 10.0f};
        std::array<float, 4> rotate = {0.0f, 140.0f, 140.0f, 77.0f};
        std::array<float, 4> rotate_v = {0.0f, 0.0f, 0.0f, 0.0f};
        std::array<int, 4> branches = {1, 50, 30, 10};
        std::array<float, 4> length = {1.0f, 0.3f, 0.6f, 0.0f};
        
        std::array<float, 4> seg_splits = {0.0f, 0.0f, 0.0f, 0.0f};
        std::array<float, 4> split_angle = {40.0f, 0.0f, 0.0f, 0.0f};
        std::array<float, 4> split_angle_v = {5.0f, 0.0f, 0.0f, 0.0f};
        std::array<int, 4> bevel_res = {10, 10, 10, 10};
        std::array<int, 4> curve_res = {5, 5, 3, 1};
        std::array<float, 4> curve = {0.0f, -40.0f, -40.0f, 0.0f};
        std::array<float, 4> curve_back = {0.0f, 0.0f, 0.0f, 0.0f};
        std::array<float, 4> curve_v = {20.0f, 50.0f, 75.0f, 0.0f};
        std::array<float, 4> bend_v = {0.0f, 50.0f, 0.0f, 0.0f};
        std::array<float, 4> branch_dist = {0.0f, 0.0f, 0.0f, 0.0f};
        std::array<float, 4> radius_mod = {1.0f, 1.0f, 1.0f, 1.0f};
        std::array<float, 4> length_v = {0.0f, 0.0f, 0.0f, 0.0f};
        std::array<float, 4> taper = {1.0f, 1.0f, 1.0f, 1.0f};
        
        // Leaf and blossom properties
        int leaves = 25;  // Number of leaves per tip
        int leaf_blos_num = 40;
        int leaf_shape = 0;
        float leaf_scale = 0.17f;
        float leaf_scale_x = 1.0f;
        float leaf_bend = 0.6f;
        int blossom_shape = 1;
        float blossom_scale = 0.0f;
        float blossom_rate = 0.0f;
        
        // Environmental effects
        std::array<float, 3> tropism = {0.0f, 0.5f, 0.0f};
        float prune_ratio = 0.0f;
        float prune_width = 0.5f;
        float prune_width_peak = 0.5f;
        float prune_power_low = 0.5f;
        float prune_power_high = 0.5f;
        
        // Additional properties (legacy)
        float gravity = 0.0f;
        float coverage = 0.0f;
        float vertically = 0.0f;
        float twist = 0.0f;
        bool trunk_mode = false;  // Monopodial or Dichotomous
    };

    class Tree {
    public:
        Tree() : m_seed(12345) {}
        Tree(uint32_t seed) : m_seed(seed) {}

        void makeStem(
            CHTurtle& turtle,
            Stem& stem,
            int start = 0,
            float split_corr_angle = 0,
            float num_branches_factor = 1,
            float clone_prob = 1,
            CHTurtle* pos_corr_turtle = nullptr, 
            CHTurtle* cloned_turtle = nullptr);
        void makeBranches(
            CHTurtle& turtle, 
            Stem& stem, 
            int seg_ind, 
            int branch_on_seg, 
            std::array<float, 1>& prev_rotation_angle, 
            bool is_leaves = false);
        void makeClones(
            CHTurtle& turtle,
            int seg_ind, float split_corr_angle,
            float num_branches_factor,
            float clone_prob,
            Stem& stem,
            int num_of_splits,
            float spl_angle,
            float spr_angle,
            bool is_base_split
        );
        bool testStem(CHTurtle& turtle, Stem& stem, float start, float split_corr_angle, float clone_prob);
        void createBranches();

        // Tree generation state
        void setSeed(uint32_t seed) { m_seed = seed; }
        uint32_t getSeed() const { return m_seed; }

        const TreeParam& getParams() const { return m_params; }
        TreeParam& getParams() { return m_params; }

        const std::vector<Stem>& getStems() const { return m_stems; }
        const std::vector<Leaf>& getLeaves() const { return m_leaves; }
        const std::vector<std::shared_ptr<BezierCurve>>& getBranchCurves() const { return m_branch_curves; }

    private:
        // Helper functions
        float calcStemLength(const Stem& stem);
        float calcStemRadius(const Stem& stem);
        float radiusAtOffset(const Stem& stem, float offset);
        void applyTropism(CHTurtle& turtle, const Vec3f& tropism_v);
        float calcLeafCount(const Stem& stem);
        float calcBranchCount(const Stem& stem);
        float calcCurveAngle(int depth, int seg_ind);
        float calcRotateAngle(int depth, float prev_angle);
        float calcDownAngle(Stem& stem, float stem_offset);
        std::tuple<Vec3f, Vec3f, Vec3f, Vec3f> calcHelixPoints(const CHTurtle& turtle, float rad, float pitch);
        void increaseBezierPointRes(Stem& stem, int seg_ind, int points_per_seg);
        void scaleBezierHandlesForFlare(Stem& stem, int max_points_per_seg);
        bool pointInside(const Vec3f& point);
        float shapeRatio(int shape, float ratio);
        std::tuple<CHTurtle, CHTurtle, float, float> setupBranch(
            CHTurtle& turtle, Stem& stem, BranchMode branch_mode, float offset,
            BezierPoint* start_point, BezierPoint* end_point, float stem_offset,
            int branch_ind, std::array<float, 1>& prev_rot_ang, int branches_in_group = 0
        );
        CHTurtle makeBranchDirTurtle(CHTurtle& turtle, bool helix, float offset, BezierPoint* start_point, BezierPoint* end_point);
        CHTurtle makeBranchPosTurtle(CHTurtle& dir_turtle, float offset, BezierPoint* start_point, BezierPoint* end_point, float radius_limit);
        
        TreeParam m_params;
        std::vector<Stem> m_stems;
        std::vector<Leaf> m_leaves;
        uint32_t m_seed;

        std::vector<std::shared_ptr<BezierCurve>> m_branch_curves;
        
        // Tree generation state
        int m_stem_index = 0;
        float m_base_length = 0.0f;
        float m_tree_scale = 1.0f;
        float m_trunk_length = 0.0f;
        // Flag to generate leaves or not
        bool m_generate_leaves = true;
        std::array<int, 7> m_split_num_error = { 0, 0, 0, 0, 0, 0, 0 };
    };

    enum class TreeCharacter {
        EXCLAMATION,        // Set turtle width to w
        F,                  // Move turtle forward by length l, and draw branch
        A,                  // Close end of branch, i.e., taper to 0 radius
        PLUS,               // Rotate left by angle a
        MINUS,              // Rotate right by angle a
        AND,                // Pitch turtle down by angle a
        CIRCUMFLEX,         // Pitch up by angle a
        BACKSLASH,          // Roll left by angle a
        FORWARD_SLASH,      // Roll right by angle a
        L,                  // Create leaf according to d_ang and r_ang
        LBRACKET,           // Start branch
        RBRACKET,           // End branch
        DOLLAR              // Save current turtle state
    };
#endif

    // L-system tree segment structure (with orientation basis)
    struct TreeSegment {
        Vec3f position;     // Segment position
        Vec3f direction;    // Direction (normalized)
        Vec3f normal;       // Normal vector
        Vec3f binormal;     // Binormal vector
        float radius;       // Radius
        int generation;     // Generation number
        
        // Default basis creation (Onb auto-generates orthonormal basis)
        TreeSegment(Vec3f pos, Vec3f dir, float r, int gen) 
            : position(pos), direction(normalize(dir)), radius(r), generation(gen) {
            // Onb generates orthonormal basis perpendicular to direction
            Onb frame(direction);
            normal = frame.tangent;
            binormal = frame.bitangent;
        }
        
        // Inherit parent basis for smooth connection (project to be perpendicular to direction)
        TreeSegment(Vec3f pos, Vec3f dir, float r, int gen, Vec3f parent_normal, Vec3f parent_binormal) 
            : position(pos), direction(normalize(dir)), radius(r), generation(gen) {
            // Project parent normal to be perpendicular to direction (Gram-Schmidt)
            normal = normalize(parent_normal - dot(parent_normal, direction) * direction);
            
            // Binormal is cross product of direction and normal (right-hand system)
            binormal = normalize(cross(direction, normal));
            
            // Safety check: regenerate with Onb if normal degenerates
            if (length(normal) < 0.1f) {
                Onb frame(direction);
                normal = frame.tangent;
                binormal = frame.bitangent;
            }
        }
    };

#ifndef __CUDACC__
    // CPU tree structure (recursive, using shared_ptr)
    struct TreeBranch {
        std::vector<TreeSegment> segments;
        std::vector<std::shared_ptr<TreeBranch>> children;
        int generation;
        float length;
        float radius;
        
        TreeBranch(int gen, float len, float r) : generation(gen), length(len), radius(r) {}
        
        void addChild(std::shared_ptr<TreeBranch> child) {
            children.push_back(child);
        }
    };
#endif

    // CUDA tree structure (flat array, index-based)
    struct CudaTreeSegment {
        Vec3f position;
        Vec3f direction;
        Vec3f normal;
        Vec3f binormal;
        float radius;
        int generation;
    };

    struct CudaTreeBranch {
        int segment_start;      // Start index in segments array
        int segment_count;      // Number of segments in this branch
        int child_start;        // Start index in children array
        int child_count;        // Number of child branches
        int parent_idx;         // Parent branch index (-1 = root)
        int generation;
        float length;
        float radius;
        bool is_terminal;       // Is terminal branch
    };

    // Mesh construction output buffers
    struct TreeMeshBuffers {
        Vec3f* vertices;
        Vec3f* normals;
        Vec2f* texcoords;
        Vec3i* face_indices;
        Vec3i* normal_indices;
        Vec3i* texcoord_indices;
        
        int* vertex_count;      // Current vertex count
        int* face_count;        // Current face count
        int max_vertices;       // Maximum vertices
        int max_faces;          // Maximum faces
    };

} // namespace prayground
