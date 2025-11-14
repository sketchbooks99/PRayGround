#include "tree.h"
#include <prayground/math/util.h>
#include <prayground/math/random.h>

namespace prayground {

    // ------------------------------------------------------------------------------
    // Leaf
    // ------------------------------------------------------------------------------
    MeshData Leaf::getShape(float g_scale, float scale, float scale_x)
    {
        MeshData mesh_data;

        auto _getVertex = [&](const Vec3f& v) {
            Vec3f vertex = Vec3f(
                v.x() * scale * scale_x * g_scale,
                v.y() * scale * g_scale,
                v.z() * scale * g_scale
            );
            return vertex;

            };

        Vec3f v0 = _getVertex(Vec3f(-1.0f, 0.0f, 0.0f));
        Vec3f v1 = _getVertex(Vec3f(1.0f, 0.0f, 0.0f));
        Vec3f v2 = _getVertex(Vec3f(0.0f, 1.0f, 0.0f));
        Vec3f v3 = _getVertex(Vec3f(0.0f, -1.0f, 0.0f));

        Vec2f uv0(1, 0);
        Vec2f uv1(1, 1);
        Vec2f uv2(0, 1);
        Vec2f uv3(0, 0);

        Vec3f n = Vec3f(0.0f, 0.0f, 1.0f);

        int idx0 = (int)mesh_data.vertices.size();
        int idx1 = idx0 + 1;
        int idx2 = idx0 + 2;
        int idx3 = idx0 + 3;

        Face f0 = {
            .vertex_id = Vec3i(idx0, idx1, idx2),
            .normal_id = Vec3i(idx0, idx1, idx2),
            .texcoord_id = Vec3i(idx0, idx1, idx2)
        };
        Face f1 = {
            .vertex_id = Vec3i(idx0, idx2, idx3),
            .normal_id = Vec3i(idx0, idx2, idx3),
            .texcoord_id = Vec3i(idx0, idx2, idx3)
        };

        mesh_data.vertices.push_back(v0);
        mesh_data.vertices.push_back(v1);
        mesh_data.vertices.push_back(v2);
        mesh_data.vertices.push_back(v3);

        mesh_data.normals.push_back(n);
        mesh_data.normals.push_back(n);
        mesh_data.normals.push_back(n);
        mesh_data.normals.push_back(n);

        mesh_data.texcoords.push_back(uv0);
        mesh_data.texcoords.push_back(uv1);
        mesh_data.texcoords.push_back(uv2);
        mesh_data.texcoords.push_back(uv3);

        std::vector<Face> face_group;
        face_group.push_back(f0);
        face_group.push_back(f1);
        mesh_data.faces.push_back(face_group);

        return mesh_data;
    }

    void Tree::makeClones(
        CHTurtle& turtle,
        int seg_ind, float split_corr_angle,
        float num_branches_factor,
        float clone_prob,
        Stem& stem,
        int num_of_splits,
        float spl_angle,
        float spr_angle,
        bool is_base_split
    )
    {
        // Make clones of branch used if seg_splits or base_splits > 0  

        bool using_direct_split = m_params.split_angle[stem.depth] < 0;
        float stem_depth = m_params.split_angle_v[stem.depth];

        // Validate split count for direct split mode  
        if (!is_base_split && num_of_splits > 2 && using_direct_split) {
            throw std::runtime_error("Only splitting up to 3 branches is supported");
        }

        for (int split_index = 0; split_index < num_of_splits; split_index++) {
            // Copy turtle for new branch  
            CHTurtle n_turtle(turtle);

            // Tip branch down away from axis of stem  
            n_turtle.pitchDown(math::radians(spl_angle / 2));  // Convert degrees to radians

            // Spread out clones  
            float eff_spr_angle;
            if (is_base_split && !using_direct_split) {
                // Base splits: distribute evenly around circle  
                eff_spr_angle = (split_index + 1) * (360.0 / (num_of_splits + 1)) +
                    rnd(m_seed, -1.0f, 1.0f) * stem_depth;
            }
            else {
                // Regular splits: alternate sides  
                if (split_index == 0) {
                    eff_spr_angle = spr_angle / 2;
                }
                else {
                    eff_spr_angle = -spr_angle / 2;
                }
            }

            // Apply spread angle  
            if (using_direct_split) {
                n_turtle.turnLeft(math::radians(eff_spr_angle));  // Convert degrees to radians
            }
            else {
                Quaternion quat(Vec3f(0, 0, 1), math::radians(eff_spr_angle));

                n_turtle.setDirection(quat.rotate(n_turtle.direction()));
                turtle.setDirection(normalize(turtle.direction()));

                n_turtle.setRight(quat.rotate(n_turtle.right()));
                turtle.setRight(normalize(turtle.right()));
            }

            // Create new clone branch and set up then recurse  
            std::shared_ptr<BezierSpline> split_stem = m_branch_curves[stem.depth]->addSpline();
            split_stem->resolution_u = stem.curve->resolution_u;
            split_stem->radius_interpolation = RadiusInterpolation::CARDINAL;

            Stem new_stem = stem.copy();
            new_stem.curve = split_stem;

            // Determine if cloned turtle should be passed  
            CHTurtle* cloned = nullptr;
            if (m_params.split_angle_v[stem.depth] >= 0) {
                cloned = &turtle;
            }

            // Recursively create the clone branch  
            makeStem(n_turtle, new_stem, seg_ind, split_corr_angle,
                num_branches_factor, clone_prob, cloned);
        }
    }

    MeshData Leaf::getMesh(float bend, const MeshData& mesh, int offset) const
    {
        MeshData result;
        Quatf trf = Quatf::trackTo(m_dir, Vec3f(0, 0, 1), Vec3f(0, 1, 0));

        Vec3f right_transformed = trf.inverse().rotate(m_right);
        float spin_angle = math::pi - angle(right_transformed, Vec3f(1, 0, 0));
        Quatf spin_quat(Vec3f(0, 0, 1), spin_angle);

        // If bend is needed
        Quatf bend_quat1 = Quatf::identity();
        if (bend > 0.0f) {
            auto [bend_trf1, bend_trf2] = calcBendTransform(bend);
            bend_quat1 = bend_trf1;
        }

        // Transform vertices
        for (const auto& v : mesh.vertices) {
            Vec3f new_vertex = v;
            new_vertex = spin_quat.rotate(new_vertex);
            new_vertex = trf.rotate(new_vertex);
            if (bend > 0.0f) {
                new_vertex = bend_quat1.rotate(new_vertex);
            }
            new_vertex += m_pos;
            result.vertices.push_back(new_vertex);
        }

        // Transform normals
        for (const auto& n : mesh.normals) {
            Vec3f new_normal = n;
            new_normal = spin_quat.rotate(new_normal);
            new_normal = trf.rotate(new_normal);
            if (bend > 0.0f) {
                new_normal = bend_quat1.rotate(new_normal);
            }
            result.normals.push_back(normalize(new_normal));
        }
        result.texcoords = mesh.texcoords;
        result.faces = mesh.faces;

        return result;
    }

    std::pair<Quatf, Quatf> Leaf::calcBendTransform(float bend) const
    {
        Vec3f normal = cross(m_dir, m_right);

        float theta_pos = atan2(m_pos.y(), m_pos.x());
        float theta_normal = atan2(normal.y(), normal.x());
        float theta_bend = theta_pos - theta_normal;

        float angle1 = theta_bend * bend;
        /*Matrix4f bend_trf1 = Matrix4f::rotate(angle1, Vec3f(0, 0, 1));*/
        Quatf bend_trf1(Vec3f(0, 0, 1), angle1);

        Vec3f new_dir = bend_trf1.rotate(m_dir);
        Vec3f new_right = bend_trf1.rotate(m_right);
        Vec3f new_normal = cross(new_dir, new_right);

        float phi_bend = declination(new_normal);

        if (phi_bend > math::pi / 2.0f)
            phi_bend -= math::pi;

        float angle2 = phi_bend * bend;
        Quatf bend_trf2(new_right, phi_bend * bend);

        return { bend_trf1, bend_trf2 };
    }

    // ------------------------------------------------------------------------------
    // CHTurtle
    // ------------------------------------------------------------------------------
    CHTurtle::CHTurtle()
        : m_pos(Vec3f(0.0f)), m_dir(Vec3f(0.0f, 0.0f, 1.0f)), m_right(Vec3f(1.0f, 0.0f, 0.0f)), m_width(1.0f)
    {
    }

    CHTurtle::CHTurtle(const Vec3f& pos, const Vec3f& dir, const Vec3f& right, float width)
        : m_pos(pos), m_dir(dir), m_right(right), m_width(width)
    {

    }

    CHTurtle::CHTurtle(const CHTurtle& other)
        : m_pos(other.m_pos), m_dir(other.m_dir), m_right(other.m_right), m_width(other.m_width)
    {

    }

    void CHTurtle::move(float distance)
    {
        m_pos += m_dir * distance;
    }

    void CHTurtle::turnRight(float angle)
    {
        // Rotate around up vector (perpendicular to dir and right)
        Vec3f up = cross(m_dir, m_right);
        Quatf q(up, angle);
        m_dir = q.rotate(m_dir);
        m_right = q.rotate(m_right);
        m_dir = normalize(m_dir);
        m_right = normalize(m_right);
    }

    void CHTurtle::turnLeft(float angle)
    {
        turnRight(-angle);
    }

    void CHTurtle::pitchUp(float angle)
    {
        // Rotate around right vector
        Quatf q(m_right, angle);
        m_dir = q.rotate(m_dir);
        m_dir = normalize(m_dir);
    }

    void CHTurtle::pitchDown(float angle)
    {
        pitchUp(-angle);
    }

    void CHTurtle::rollRight(float angle)
    {
        // Rotate around direction vector
        Quatf q(m_dir, angle);
        m_right = q.rotate(m_right);
        m_right = normalize(m_right);
    }

    void CHTurtle::rollLeft(float angle)
    {
        rollRight(-angle);
    }

    // ------------------------------------------------------------------------------
    // Tree
    // ------------------------------------------------------------------------------
    
    // ----------------------------------------------------------------------------------------------------------------------------
    // Calculate stem length based on parent and parameters
    float Tree::calcStemLength(const Stem& stem) {
        float result = 0.0f;
        if (stem.depth == 0) {
            result = m_tree_scale * (m_params.length[0] + rnd(m_seed, -1.0f, 1.0f) * m_params.length_v[0]);
            m_trunk_length = result;
        }
        else if (stem.depth == 1) {
            float ratio = (stem.parent->length - stem.offset) / (stem.parent->length - m_base_length);
            float shape_result = shapeRatio(m_params.shape, ratio);
            result = stem.parent->length * stem.parent->length_child_max * shape_result;
        }
        else {
            // For depth >= 2, use a simpler calculation based on parent length and offset
            // Use the distance from attachment point to tip as a scaling factor
            float remaining_parent = stem.parent->length - stem.offset;
            result = stem.parent->length_child_max * remaining_parent;
        }
        return max(0.0f, result);
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    // Calculate stem radius based on length and parameters
    float Tree::calcStemRadius(const Stem& stem) {
        float result = 0.0f;
        if (stem.depth == 0.0f) {
            result = stem.length * m_params.ratio * m_params.radius_mod[0];
        }
        else {
            result = m_params.radius_mod[stem.depth] * stem.parent->radius * pow((stem.length / stem.parent->length), m_params.ratio_power);
            result = max(0.005f, result);
            result = min(stem.radius_limit, result);
        }
        return result;
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    // Get radius at specific offset along stem (0 = base, 1 = tip)
    float Tree::radiusAtOffset(const Stem& stem, float offset) {
        int depth = stem.depth;
        float taper = m_params.taper[depth];
        
        float baseRadius = stem.radius;
        float tipRadius = baseRadius * (1.0f - offset);
        
        // Apply taper curve
        if (taper < 1.0f) {
            // Taper < 1: more cylindrical
            tipRadius = baseRadius * (1.0f - offset * taper);
        } else if (taper > 1.0f && taper < 3.0f) {
            // Taper > 1: more conical
            float taperFactor = pow(1.0f - offset, 2.0f / taper);
            tipRadius = baseRadius * taperFactor;
        }
        
        return max(tipRadius, 0.0001f);
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    void Tree::applyTropism(CHTurtle& turtle, const Vec3f& tropism_v)
    {
        Vec3f h_cross_T = cross(turtle.direction(), tropism_v);

        float alpha = 10.0f * length(h_cross_T);
        h_cross_T = normalize(h_cross_T);

        Quaternion rot_quat(h_cross_T, math::radians(alpha));

        turtle.setDirection(normalize(rot_quat.rotate(turtle.direction())));
        turtle.setRight(normalize(rot_quat.rotate(turtle.right())));
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    float Tree::calcLeafCount(const Stem& stem)
    {
        if (m_params.leaf_blos_num >= 0) {
            float leaves = m_params.leaf_blos_num * m_tree_scale / m_params.g_scale;
            
            if (!stem.parent) {
                return 0.0f;
            }
            
            float ratio = stem.length / (stem.parent->length_child_max * stem.parent->length);
            float result = leaves * ratio;
            
            return result;
        }
        else 
            return m_params.leaf_blos_num;
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    float Tree::calcBranchCount(const Stem& stem)
    {
        int d_p_1 = min(stem.depth + 1, 3);
        float result;

        if (stem.depth == 0)
            result = m_params.branches[d_p_1] * (rnd(m_seed) * 0.2f + 0.9f);
        else {
            if (m_params.branches[d_p_1] < 0) {
                result = m_params.branches[d_p_1];
            }
            else if (stem.depth == 1) {
                result = m_params.branches[d_p_1] * (0.2f + 0.8f * 
                    (stem.length / stem.parent->length) / stem.parent->length_child_max);
            }
            else {
                result = m_params.branches[d_p_1] * (1.0f - 0.5f * stem.offset / stem.parent->length);
            }
        }

        return result / (1.0f - m_params.base_size[stem.depth]);
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    float Tree::calcCurveAngle(int depth, int seg_ind)
    {
        float curve = m_params.curve[depth];
        float curve_v = m_params.curve_v[depth];
        float curve_back = m_params.curve_back[depth];
        int curve_res = static_cast<int>(m_params.curve_res[depth]);

        float curve_angle;
        if (curve_back == 0) {
            curve_angle = curve / curve_res;
        }
        else {
            if (seg_ind < curve_res / 2.0f) {
                curve_angle = curve / (curve_res / 2.0f);
            }
            else {
                curve_angle = curve_back / (curve_res / 2.0f);
            }
        }

        curve_angle += rnd(m_seed, -1.0f, 1.0f) * (curve_v / curve_res);
        return math::radians(curve_angle);  // Convert degrees to radians
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    float Tree::calcRotateAngle(int depth, float prev_angle)
    {
        // Calculate rotate angle as defined in paper, limit to 0-360
        // prev_angle is in radians, convert to degrees for calculation
        float prev_angle_deg = math::degrees(prev_angle);
        float r_angle;

        if (m_params.rotate[depth] >= 0) {
            r_angle = std::fmod(prev_angle_deg + m_params.rotate[depth] +
                rnd(m_seed, -1.0f, 1.0f) * m_params.rotate_v[depth], 360.0f);
        }
        else {
            r_angle = prev_angle_deg * (180 + m_params.rotate[depth] +
                rnd(m_seed, -1.0f, 1.0f) * m_params.rotate_v[depth]);
        }

        return math::radians(r_angle);  // Convert degrees to radians
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    float Tree::calcDownAngle(Stem& stem, float stem_offset)
    {
        // Calculate down angle as defined in paper  
        int d_plus_1 = min(stem.depth + 1, 3);
        float d_angle;

        if (m_params.down_angle_v[d_plus_1] >= 0) {
            // Normal variation mode  
            d_angle = m_params.down_angle[d_plus_1] +
                rnd(m_seed, -1.0f, 1.0f) * m_params.down_angle_v[d_plus_1];
        }
        else {
            // Distribution mode - angle varies along parent stem  
            float ratio = (stem.length - stem_offset) /
                (stem.length * (1.0f - m_params.base_size[stem.depth]));
            d_angle = m_params.down_angle[d_plus_1] +
                (m_params.down_angle_v[d_plus_1] *
                    (1.0f - 2.0f * shapeRatio(8, ratio)));

            // Introduce some variance to improve visual result  
            d_angle += rnd(m_seed, -1.0f, 1.0f) * fabsf(d_angle * 0.1);
        }

        return math::radians(d_angle);  // Convert degrees to radians
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    std::tuple<Vec3f, Vec3f, Vec3f, Vec3f> Tree::calcHelixPoints(const CHTurtle& turtle, float rad, float pitch)
    {
        std::vector<Vec3f> points = {
            Vec3f(0.0f, -rad, -pitch / 4.0f),
            Vec3f((4.0f * rad) / 3.0f, -rad, 0.0f),
            Vec3f((4.0f * rad) / 3.0f, rad, 0.0f),
            Vec3f(0, rad, pitch / 4.0f)
        };

        Quatf trf = Quatf::trackTo(turtle.direction(), Vec3f(0, 0, 1), Vec3f(0, 1, 0));
        float spin_ang = rnd(m_seed, 0, math::two_pi);
        Quatf rot_quat(Vec3f(0, 0, 1), spin_ang);

        for (auto& p : points) {
            p = rot_quat.rotate(p);
            p = trf.rotate(p);
        }

        return std::make_tuple(
            points[1] - points[0],
            points[2] - points[0],
            points[3] - points[0],
            turtle.direction()
        );
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    void Tree::increaseBezierPointRes(Stem& stem, int seg_ind, int points_per_seg)
    {
        int curve_res = static_cast<int>(m_params.curve_res[stem.depth]);
        BezierPoint* seg_end_point = &stem.curve->bezier_points.back();

        BezierPoint end_point(*seg_end_point);

        // SAFETY CHECK: Ensure we have at least 2 points before accessing [size-2]
        if (stem.curve->bezier_points.size() < 2) {
            pgLog("[ERROR] splitStem: bezier_points size=" + std::to_string(stem.curve->bezier_points.size()) + 
                  " is too small (need at least 2)\n");
            return;
        }

        BezierPoint* seg_start_point = &stem.curve->bezier_points[stem.curve->bezier_points.size() - 2];
        BezierPoint start_point(*seg_start_point);

        for (int k = 0; k < points_per_seg; k++) {
            float offset = k / static_cast<float>(points_per_seg - 1);
            BezierPoint* curr_point;

            if (k == 0) {
                curr_point = seg_start_point;
            }
            else {
                if (k == 1) {
                    curr_point = seg_end_point;
                }
                else {
                    curr_point = stem.curve->addBezierPoint();
                }

                if (k == points_per_seg - 1) {
                    curr_point->co = end_point.co;
                    curr_point->handle_left = end_point.handle_left;
                    curr_point->handle_right = end_point.handle_right;
                }
                else {
                    curr_point->co = BezierSpline::evaluateCubicBezier(offset, start_point, end_point);
                    Vec3f tangent = normalize(BezierSpline::evaluateCubicBezierTangent(offset, start_point, end_point));
                    float dir_vec_mag = pow2(length((end_point.handle_left - end_point.co)));
                    curr_point->handle_left = curr_point->co - tangent * dir_vec_mag;
                    curr_point->handle_right = curr_point->co + tangent * dir_vec_mag;
                }
            }

            curr_point->radius = radiusAtOffset(stem, (offset + seg_ind - 1) / curve_res);
        }
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    void Tree::scaleBezierHandlesForFlare(Stem& stem, int max_points_per_seg)
    {
        for (auto& point : stem.curve->bezier_points) {
            point.handle_left = point.co + (point.handle_left - point.co) / max_points_per_seg;
            point.handle_right = point.co + (point.handle_right - point.co) / max_points_per_seg;
        }
    }

    bool Tree::pointInside(const Vec3f& point)
    {
        float dist = sqrtf(pow2(point.x()) + pow2(point.y()));
        float ratio = (m_tree_scale - point.z()) / (m_tree_scale * (1.0f - m_params.base_size[0]));
        bool inside = (dist / m_tree_scale) < (m_params.prune_width * shapeRatio(8, ratio));
        return inside;
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    float Tree::shapeRatio(int shape, float ratio)
    {
        // Conical
        float result = 0.2f + 0.8f * ratio;
        
        // Envelope
        // Spherical
        switch (shape) {
        case 1:
            result = 0.2f + 0.8f * sinf(math::pi * ratio);
            break;
        case 2:
            result = 0.2f + 0.8f * sinf(0.5f * math::pi * ratio);
            break;
        case 3:
            result = 1.0f;
            break;
        case 4:
            result = 0.5f + 0.5f * ratio;
            break;
        case 5:
            if (ratio <= 0.7f)
                result = ratio / 0.7f;
            else
                result = 0.5f + 0.5f * (1.0f - ratio) / 0.3f;
            break;
        case 6:
            result = 1.0f - 0.8f * ratio;
            break;
        case 7:
            if (ratio <= 0.7f)
                result = 0.5f + 0.5f * ratio / 0.7f;
            else
                result = 0.5f + 0.5f * (1.0f - ratio) / 0.3f;
            break;
        case 8:
            if (ratio < 0 || ratio > 1) {
                result = 0.0f;
            }
            else if (ratio < 1 - m_params.prune_width_peak) {
                result = pow(ratio / (1.0f - m_params.prune_width_peak), m_params.prune_power_high);
            }
            else {
                result = pow((1.0f - ratio) / (1.0f - m_params.prune_width_peak), m_params.prune_power_low);
            }
            break;
        default:
            result = 0.2f + 0.8f * ratio;
        }

        return result;
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    std::tuple<CHTurtle, CHTurtle, float, float> Tree::setupBranch(CHTurtle& turtle, Stem& stem, BranchMode branch_mode, float offset, BezierPoint* start_point, BezierPoint* end_point, float stem_offset, int branch_ind, std::array<float, 1>& prev_rot_ang, int branches_in_group)
    {
        // Set up a new branch, creating the new direction and position turtle  
    // and orienting them correctly  
        int d_plus_1 = std::min(3, stem.depth + 1);

        // Make branch direction turtle  
        CHTurtle branch_dir_turtle = makeBranchDirTurtle(
            turtle,
            m_params.curve_v[stem.depth] < 0,
            offset,
            start_point,
            end_point
        );

        // Calc rotation angle  
        float radius_limit;
        if (branch_mode == BranchMode::fan) {
            float t_angle;
            if (branches_in_group == 1) {
                t_angle = 0;
            }
            else {
                t_angle = (m_params.rotate[d_plus_1] *
                    ((branch_ind / static_cast<float>(branches_in_group - 1)) - 0.5)) +
                    rnd(m_seed, -1.0f, 1.0f) * m_params.rotate_v[d_plus_1];
            }
            branch_dir_turtle.turnRight(math::radians(t_angle));  // Convert degrees to radians
            radius_limit = 0;
        }
        else {
            float r_angle;
            if (branch_mode == BranchMode::whorled) {
                // prev_rot_ang is now in radians, convert degree calculations to radians
                r_angle = prev_rot_ang[0] +
                    math::radians(360.0f * branch_ind / static_cast<float>(branches_in_group)) +
                    math::radians(rnd(m_seed, -1.0f, 1.0f) * m_params.rotate_v[d_plus_1]);
            }
            else {
                r_angle = calcRotateAngle(d_plus_1, prev_rot_ang[0]);  // Already returns radians
                if (m_params.rotate[d_plus_1] >= 0) {
                    prev_rot_ang[0] = r_angle;
                }
                else {
                    prev_rot_ang[0] = -prev_rot_ang[0];
                }
            }

            // Orient direction turtle to correct rotation  
            branch_dir_turtle.rollRight(r_angle);
            radius_limit = radiusAtOffset(stem, stem_offset / stem.length);
        }

        // Make branch position turtle in appropriate position on circumference  
        CHTurtle branch_pos_turtle = makeBranchPosTurtle(
            branch_dir_turtle,
            offset,
            start_point,
            end_point,
            radius_limit
        );

        // Calc down angle  
        float d_angle = calcDownAngle(stem, stem_offset);

        // Orient direction turtle to correct declination  
        branch_dir_turtle.pitchDown(d_angle);

        // Return branch info
        return std::make_tuple(branch_pos_turtle, branch_dir_turtle, radius_limit, stem_offset);
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    CHTurtle Tree::makeBranchDirTurtle(CHTurtle& turtle, bool helix, float offset, BezierPoint* start_point, BezierPoint* end_point)
    {
        CHTurtle branch_dir_turtle;
        Vec3f tangent = BezierSpline::evaluateCubicBezierTangent(offset, *start_point, *end_point);
        tangent = normalize(tangent);
        branch_dir_turtle.setDirection(tangent);

        if (helix) {
            // Approximation to actual normal to preserve for helix  
            Vec3f tan_d = normalize(BezierSpline::evaluateCubicBezierTangent(offset + 0.0001, *start_point, *end_point));
            branch_dir_turtle.setRight(cross(branch_dir_turtle.direction(), tan_d));
        }
        else {
            // Generally curve lines in plane defined by turtle.right  
            branch_dir_turtle.setRight(cross(cross(turtle.direction(), turtle.right()), branch_dir_turtle.direction()));
        }

        return branch_dir_turtle;
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    CHTurtle Tree::makeBranchPosTurtle(CHTurtle& dir_turtle, float offset, BezierPoint* start_point, BezierPoint* end_point, float radius_limit)
    {
        // Create and setup the turtle for the position of a new branch  
        dir_turtle.setPosition(BezierSpline::evaluateCubicBezier(offset, *start_point, *end_point));
        CHTurtle branch_pos_turtle(dir_turtle);
        branch_pos_turtle.pitchDown(math::radians(90.0f));  // Convert 90 degrees to radians
        branch_pos_turtle.move(radius_limit * 0.5f);  // Move only half the radius to stay closer to center

        return branch_pos_turtle;
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    void Tree::makeStem(
        CHTurtle& turtle, 
        Stem& stem, 
        int start, 
        float split_corr_angle,
        float num_branches_factor,
        float clone_prob,
        CHTurtle* pos_corr_turtle,
        CHTurtle* cloned_turtle)
    {
        if (stem.radius_limit >= 0 && stem.radius_limit < 1e-3f)
            return;
        
        int depth = stem.depth;
        int d_plus1 = min(depth + 1, 3);
        
        // Calculate length and radius if starting from beginning
        if (start == 0) {
            float length_param = m_params.length[d_plus1];
            float length_v_param = m_params.length_v[d_plus1];
            float random_factor = rnd(m_seed, -1.0f, 1.0f);
            
            stem.length_child_max = length_param + random_factor * length_v_param;
            stem.length = calcStemLength(stem);
            stem.radius = calcStemRadius(stem);
            
            if (depth == 0) {
                m_base_length = stem.length * m_params.base_size[0];
            }
        }

        // Correct position
        if (pos_corr_turtle != nullptr) {
            pos_corr_turtle->move(min(stem.radius, stem.radius_limit));
            turtle.setPosition(pos_corr_turtle->position());
        }

        // Pruning
        if (!cloned_turtle && m_params.prune_ratio > 0) {
            float start_length = stem.length;
            auto split_err_state = m_split_num_error;

            CHTurtle test_turtle(turtle);
            bool in_pruning_envelope = testStem(test_turtle, stem, start, split_corr_angle, clone_prob);
            while (!in_pruning_envelope) {
                stem.length *= 0.9f;
                if (stem.length < 0.15f * start_length) {
                    if (m_params.prune_ratio < 1.0f) {
                        stem.length = 0;
                        break;
                    } else {
                        return;
                    }
                }

                test_turtle = CHTurtle(turtle);
                in_pruning_envelope = testStem(test_turtle, stem, start, split_corr_angle, clone_prob);
            }

            auto fitting_length = stem.length;
            // apply reduction scaled by prune ratio
            stem.length = start_length * (1.0f - m_params.prune_ratio) + fitting_length * m_params.prune_ratio;
            // recalculate stem radius for new length
            stem.radius = calcStemRadius(stem);
            m_split_num_error = split_err_state;
        }
       
        // Get parameters
        int curve_res = static_cast<int>(m_params.curve_res[depth]);
        float seg_splits = m_params.seg_splits[depth];
        float seg_length = stem.length / curve_res;
        int base_seg_ind = std::ceil(m_params.base_size[0] * m_params.curve_res[0]);

        // 葉または枝の数を計算  
        int leaf_count = 0;
        int branch_count = 0;
        if (depth == m_params.levels - 1 && depth > 0 && m_params.leaf_blos_num != 0) {
            leaf_count = static_cast<int>(calcLeafCount(stem));
            leaf_count *= 1 - static_cast<float>(start) / curve_res;
        }
        else {
            branch_count = static_cast<int>(calcBranchCount(stem));
            branch_count *= 1 - static_cast<float>(start) / curve_res;
            branch_count *= num_branches_factor;
        }

        float f_branches_on_seg = static_cast<float>(branch_count) / curve_res;
        float f_leaves_on_seg = static_cast<float>(leaf_count) / curve_res;

        // Floyd-Steinberg誤差拡散用
        // For Floyd-Steinberg error diffusion
        float branch_num_error = 0;
        float leaf_num_error = 0;

        // Initialize rotate angle
        std::array<float, 1> prev_rotation_angle = { 0.0f };
        if (m_params.rotate[d_plus1] >= 0) {
            prev_rotation_angle[0] = rnd(m_seed, 0, math::two_pi);
        }
        else {
            prev_rotation_angle[0] = 1.0f;
        }

        // Parameter calculation of helix branch
        Vec3f helP0, helP1, helP2, helAxis;
        bool is_helix = m_params.curve_v[depth] < 0;
        if (is_helix) {
            float tan_ang = tanf(math::radians(90 - fabsf(m_params.curve_v[depth])));
            float hel_pitch = 2 * stem.length / curve_res * rnd(m_seed, 0.8f, 1.2f);
            float hel_radius = 3 * hel_pitch / (16 * tan_ang) * rnd(m_seed, 0.8f, 1.2f);

            if (depth > 1) {
                applyTropism(turtle, Vec3f(m_params.tropism[0], m_params.tropism[1], m_params.tropism[2]));
            }
            else {
                applyTropism(turtle, Vec3f(m_params.tropism[0], 0, m_params.tropism[2]));
            }

            std::tie(helP0, helP1, helP2, helAxis) = calcHelixPoints(turtle, hel_radius, hel_pitch);
        }

        // セグメントごとの処理  
        int points_per_seg = (depth == 0 || m_params.taper[depth] > 1) ?
            std::ceil(max(1.0f, 100.0f / curve_res)) : 2;

        for (int seg_ind = start; seg_ind <= curve_res; seg_ind++) {
            int remaining_segs = curve_res + 1 - seg_ind;

            // ベジェ点の設定  
            BezierPoint* new_point;
            is_helix = m_params.curve_v[depth] < 0;
            if (is_helix) {
                // ヘリックス枝の処理  
                Vec3f pos = turtle.position();
                if (seg_ind == 0 && !stem.curve->bezier_points.empty()) {
                    new_point = &stem.curve->bezier_points[0];
                    new_point->co = pos;
                    new_point->handle_right = helP0 + pos;
                    new_point->handle_left = pos;
                }
                else {
                    new_point = stem.curve->addBezierPoint();
                    if (seg_ind == 1) {
                        new_point->co = helP2 + pos;
                        new_point->handle_left = helP1 + pos;
                        new_point->handle_right = 2 * new_point->co - new_point->handle_left;
                    }
                    else {
                        // SAFETY CHECK: Ensure we have at least 2 points
                        if (stem.curve->bezier_points.size() < 2) {
                            pgLog("[ERROR] makeStem helix: bezier_points size=" + 
                                  std::to_string(stem.curve->bezier_points.size()) + 
                                  " at seg_ind=" + std::to_string(seg_ind) );
                            return;
                        }
                        auto prev_point = &stem.curve->bezier_points[stem.curve->bezier_points.size() - 2];
                        new_point->co = Quaternion(helAxis, (seg_ind - 1) * math::pi).rotate(helP2);
                        new_point->co += prev_point->co;
                        Vec3f difP = Quaternion(helAxis, (seg_ind - 1) * math::pi).rotate(helP2 - helP1);
                        new_point->handle_left = new_point->co - difP;
                        new_point->handle_right = 2 * new_point->co - new_point->handle_left;
                    }
                    turtle.setPosition(new_point->co);
                    turtle.setDirection(normalize(new_point->handle_right));
                }
            }
            else {
                // 通常の曲線枝  
                if (seg_ind == start) {
                    // SAFETY CHECK: Ensure bezier_points[0] exists
                    if (stem.curve->bezier_points.empty()) {
                        // First point doesn't exist yet, create it
                        new_point = stem.curve->addBezierPoint();
                    } else {
                        new_point = &stem.curve->bezier_points[0];
                    }
                }
                else {
                    turtle.move(seg_length);
                    new_point = stem.curve->addBezierPoint();
                }

                new_point->co = turtle.position();
                if (cloned_turtle && seg_ind == start) {
                    new_point->handle_left = turtle.position() - cloned_turtle->direction() * (stem.length / (curve_res * 3));
                    new_point->handle_right = turtle.position() + cloned_turtle->direction() * (stem.length / (curve_res * 3));
                }
                else {
                    new_point->handle_left = turtle.position() - turtle.direction() * stem.length / (curve_res * 3);
                    new_point->handle_right = turtle.position() + turtle.direction() * stem.length / (curve_res * 3);
                }
            }

            // 半径の設定  
            float actual_radius = radiusAtOffset(stem, static_cast<float>(seg_ind) / curve_res);
            new_point->radius = actual_radius;

            if (seg_ind > start) {
                // 分岐数の計算  
                int num_of_splits = 0;
                if (!is_helix) {
                    if (m_params.base_splits > 0 && depth == 0 && seg_ind == base_seg_ind) {
                        num_of_splits = m_params.base_splits < 0 ?
                            static_cast<int>(rnd(m_seed) * (std::abs(m_params.base_splits) + 0.5f)) :
                            m_params.base_splits;
                    }
                    else if (seg_splits > 0 && seg_ind < curve_res &&
                        (depth > 0 || seg_ind > base_seg_ind)) {
                        if (rnd(m_seed) <= clone_prob) {
                            num_of_splits = static_cast<int>(seg_splits + m_split_num_error[depth]);
                            m_split_num_error[depth] -= num_of_splits - seg_splits;

                            clone_prob /= num_of_splits + 1;
                            num_branches_factor /= num_of_splits + 1;
                            num_branches_factor = max(0.8f, num_branches_factor);

                            branch_count *= num_branches_factor;
                            f_branches_on_seg = static_cast<float>(branch_count) / curve_res;
                        }
                    }
                }

                // 枝または葉の配置  
                if (std::abs(branch_count) > 0 && depth < m_params.levels - 1) {
                    int branches_on_seg;
                    if (branch_count < 0) {
                        branches_on_seg = (seg_ind == curve_res) ? static_cast<int>(branch_count) : 0;
                    }
                    else {
                        branches_on_seg = static_cast<int>(f_branches_on_seg + branch_num_error);
                        branch_num_error -= branches_on_seg - f_branches_on_seg;
                    }

                    if (std::abs(branches_on_seg) > 0) {
                        makeBranches(turtle, stem, seg_ind, branches_on_seg, prev_rotation_angle);
                    }
                }
                else if (std::abs(leaf_count) > 0 && depth > 0) {
                    int leaves_on_seg;
                    if (leaf_count < 0) {
                        leaves_on_seg = (seg_ind == curve_res) ? leaf_count : 0;
                    }
                    else {
                        leaves_on_seg = static_cast<int>(f_leaves_on_seg + leaf_num_error);
                        leaf_num_error -= leaves_on_seg - f_leaves_on_seg;
                    }

                    if (std::abs(leaves_on_seg) > 0) {
                        makeBranches(turtle, stem, seg_ind, leaves_on_seg, prev_rotation_angle, true);
                    }
                }

                // 分岐（split）処理  
                if (!is_helix) {
                    if (num_of_splits > 0) {
                        bool is_base_split = (m_params.base_splits > 0 && depth == 0 && seg_ind == base_seg_ind);
                        bool using_direct_split = m_params.split_angle[depth] < 0;

                        float spl_angle, spr_angle;
                        if (using_direct_split) {
                            spr_angle = std::abs(m_params.split_angle[depth]) +
                                rnd(m_seed,-1.0f, 1.0f)* m_params.split_angle_v[depth];
                            spl_angle = 0;
                            split_corr_angle = 0;
                        }
                        else {
                            float declination = turtle.direction().declination();
                            spl_angle = m_params.split_angle[depth] +
                                rnd(m_seed, -1.0f, 1.0f) * m_params.split_angle_v[depth] - declination;
                            spl_angle = std::max(0.0f, spl_angle);
                            split_corr_angle = spl_angle / remaining_segs;
                            spr_angle = -(20.0f + 0.75f * (30 + std::abs(declination - 90) *
                                pow(rnd(m_seed), 2)));
                        }

                        makeClones(turtle, seg_ind, split_corr_angle, num_branches_factor,
                            clone_prob, stem, num_of_splits, spl_angle, spr_angle, is_base_split);

                        turtle.pitchDown(math::radians(spl_angle / 2));  // Convert degrees to radians

                        if (!is_base_split && num_of_splits == 1) {
                            if (using_direct_split) {
                                turtle.turnRight(math::radians(spr_angle / 2));
                            }
                            else {
                                Quaternion quat(Vec3f(0, 0, 1), math::radians(-spr_angle / 2));
                                turtle.setDirection(normalize(quat.rotate(turtle.direction())));
                                turtle.setRight(normalize(quat.rotate(turtle.right())));
                            }
                        }
                    }
                    else {
                        turtle.turnLeft(math::radians(rnd(m_seed, -1.0f, 1.0f) * m_params.bend_v[depth] / curve_res));  // Convert degrees to radians
                        float curveAngle = calcCurveAngle(depth, seg_ind);  // Already returns radians
                        turtle.pitchDown(curveAngle - split_corr_angle);
                    }

                    if (depth > 1) {
                        applyTropism(turtle, Vec3f(m_params.tropism[0], m_params.tropism[1], m_params.tropism[2]));
                    }
                    else {
                        applyTropism(turtle, Vec3f(m_params.tropism[0], 0.0f, m_params.tropism[2]));
                    }
                }

                if (points_per_seg > 2) {
                    increaseBezierPointRes(stem, seg_ind, points_per_seg);
                }
            }
        }

        // フレア効果のためのベジェハンドルのスケーリング  
        if (points_per_seg > 2) {
            scaleBezierHandlesForFlare(stem, points_per_seg);
        }

        // Check for overly vertical branches
        if (stem.depth > 0 && stem.curve->bezier_points.size() >= 2) {
            auto& first_point = stem.curve->bezier_points[0];
            auto& last_point = stem.curve->bezier_points.back();
            Vec3f branch_direction = normalize(last_point.co - first_point.co);
            float y_component = fabs(branch_direction.y());
        }

        m_stem_index++;
    }

    // ----------------------------------------------------------------------------------------------------------------------------
    void Tree::makeBranches(
        CHTurtle& turtle, 
        Stem& stem, 
        int seg_ind, 
        int branches_on_seg, 
        std::array<float, 1>& prev_rotation_angle, 
        bool is_leaves
    )
    {
        // Make the required branches for a segment of the stem  
        // Copy bezier points to avoid dangling pointers if stem is local
        BezierPoint start_point_copy = stem.curve->bezier_points[stem.curve->bezier_points.size() - 2];
        BezierPoint end_point_copy = stem.curve->bezier_points.back();
        BezierPoint* start_point = &start_point_copy;
        BezierPoint* end_point = &end_point_copy;
        std::vector<std::tuple<CHTurtle, CHTurtle, float, float>> branches_array;
        int d_plus_1 = std::min(3, stem.depth + 1);

        if (branches_on_seg < 0) {  // fan branches  
            for (int branch_ind = 0; branch_ind < std::abs(branches_on_seg); branch_ind++) {
                float stem_offset = 1;
                branches_array.push_back(
                    setupBranch(turtle, stem, BranchMode::fan, 1, start_point, end_point,
                        stem_offset, branch_ind, prev_rotation_angle,
                        std::abs(branches_on_seg))
                );
            }
        }
        else {
            float base_length = stem.length * m_params.base_size[stem.depth];
            float branch_dist = m_params.branch_dist[d_plus_1];
            int curve_res = static_cast<int>(m_params.curve_res[stem.depth]);

            if (branch_dist > 1) {  // whorled branches  
                // Calc number of whorls  
                int num_of_whorls = static_cast<int>(branches_on_seg / (branch_dist + 1));
                float branches_per_whorl = branch_dist + 1;
                float branch_whorl_error = 0;

                for (int whorl_num = 0; whorl_num < num_of_whorls; whorl_num++) {
                    // Calc whorl offset in segment and on stem  
                    float offset = std::min(std::max(0.0f, whorl_num / static_cast<float>(num_of_whorls)), 1.0f);
                    float stem_offset = (((seg_ind - 1) + offset) / curve_res) * stem.length;

                    // If not in base area then make the branches  
                    if (stem_offset > base_length) {
                        // Calc FS corrected num of branches this whorl  
                        int branches_this_whorl = static_cast<int>(branches_per_whorl + branch_whorl_error);
                        branch_whorl_error -= branches_this_whorl - branches_per_whorl;

                        // Set up these branches  
                        for (int branch_ind = 0; branch_ind < branches_this_whorl; branch_ind++) {
                            branches_array.push_back(
                                setupBranch(turtle, stem, BranchMode::whorled, offset,
                                    start_point, end_point, stem_offset, branch_ind,
                                    prev_rotation_angle, branches_this_whorl)
                            );
                        }
                    }

                    // Rotate start angle for next whorl (convert degrees to radians)
                    prev_rotation_angle[0] += math::radians(m_params.rotate[d_plus_1]);
                }
            }
            else {  // alternating or opposite branches  
                for (int branch_ind = 0; branch_ind < branches_on_seg; branch_ind++) {
                    // Calc offset in segment and on stem  
                    float offset;
                    if (branch_ind % 2 == 0) {
                        offset = std::min(std::max(0.0f, branch_ind / static_cast<float>(branches_on_seg)), 1.0f);
                    }
                    else {
                        offset = std::min(std::max(0.0f, (branch_ind - branch_dist) / static_cast<float>(branches_on_seg)), 1.0f);
                    }

                    float stem_offset = (((seg_ind - 1) + offset) / curve_res) * stem.length;

                    // If not in base area then set up the branch  
                    if (stem_offset > base_length) {
                        branches_array.push_back(
                            setupBranch(turtle, stem, BranchMode::alt_opp, offset,
                                start_point, end_point, stem_offset, branch_ind,
                                prev_rotation_angle)
                        );
                    }
                }
            }
        }

        // Make all new branches from branches_array  
        if (is_leaves) {
            for (const auto& [pos_tur, dir_tur, rad, b_offset] : branches_array) {
                m_leaves.push_back(Leaf{ pos_tur.position(), dir_tur.direction(), dir_tur.right(), rad });
            }
        }
        else {
            for (auto& [pos_tur, dir_tur, rad, b_offset] : branches_array) {
                std::shared_ptr<BezierSpline> new_spline = m_branch_curves[d_plus_1]->addSpline();
                new_spline->resolution_u = m_params.curve_res[d_plus_1];
                new_spline->radius_interpolation = RadiusInterpolation::CARDINAL;
                
                // Create child stem as shared_ptr so it can be added to parent's children
                auto new_stem = std::make_shared<Stem>(d_plus_1, new_spline, &stem, b_offset, rad);
                
                // Add to parent's children list
                stem.children.push_back(new_stem);
                
                // Recursively build this child stem
                makeStem(dir_tur, *new_stem, 0, 0, 1, 1, &pos_tur);
            }
        }
    }
    
    // ----------------------------------------------------------------------------------------------------------------------------
    bool Tree::testStem(CHTurtle& turtle, Stem& stem, float start, float split_corr_angle, float clone_prob)
    {
        // Test if stem is inside pruning envelope  
    // Use level 3 parameters for any depth greater than this  
        int depth = stem.depth;
        int d_plus_1 = std::min(depth + 1, 3);

        // Get parameters  
        int curve_res = static_cast<int>(m_params.curve_res[depth]);
        float seg_splits = m_params.seg_splits[depth];
        float seg_length = stem.length / curve_res;

        // Calc base segment  
        int base_seg_ind = std::ceil(m_params.base_size[0] *
            static_cast<int>(m_params.curve_res[0]));

        // Decide on start rotation for branches/leaves  
        std::vector<float> prev_rotation_angle = { 0 };
        if (m_params.rotate[d_plus_1] >= 0) {
            // Start at random rotation  
            prev_rotation_angle[0] = rnd(m_seed, 0, 360);
        }
        else {
            // In this case prev_rotation_angle used as multiplier to alternate side  
            prev_rotation_angle[0] = 1;
        }

        // Calc helix parameters if needed  
        Vec3f hel_p_2, hel_axis, previous_helix_point;
        if (m_params.curve_v[depth] < 0) {
            float tan_ang = tanf(math::radians(90 - fabsf(m_params.curve_v[depth])));
            float hel_pitch = 2 * stem.length / curve_res * rnd(m_seed, 0.8f, 1.2f);
            float hel_radius = 3 * hel_pitch / (16 * tan_ang) * rnd(m_seed, 0.8f, 1.2f);

            // Apply full tropism if not trunk/main branch and horizontal tropism if is  
            if (depth > 1) {
                applyTropism(turtle, Vec3f(m_params.tropism[0], m_params.tropism[1], m_params.tropism[2]));
            }
            else {
                applyTropism(turtle, Vec3f(m_params.tropism[0], 0.0f, m_params.tropism[2]));
            }

            auto [hel_p_0, hel_p_1, hel_p_2_temp, hel_axis_temp] =
                calcHelixPoints(turtle, hel_radius, hel_pitch);
            hel_p_2 = hel_p_2_temp;
            hel_axis = hel_axis_temp;
        }

        for (int seg_ind = start; seg_ind <= curve_res; seg_ind++) {
            int remaining_segs = curve_res + 1 - seg_ind;

            // Set up next bezier point  
            if (m_params.curve_v[depth] < 0) {
                // Negative curve_v so helix branch  
                Vec3f pos = turtle.position();
                if (seg_ind == 0) {
                    turtle.setPosition(pos);
                }
                else {
                    if (seg_ind == 1) {
                        turtle.setPosition(hel_p_2 + pos);
                    }
                    else {
                        hel_p_2 = Quaternion(hel_axis, (seg_ind - 1) * math::pi).rotate(hel_p_2);
                        turtle.setPosition(hel_p_2 + previous_helix_point);
                    }
                    previous_helix_point = turtle.position();
                }
            }
            else {
                // Normal curved branch  
                // Move turtle  
                if (seg_ind != start) {
                    turtle.move(seg_length);
                    if (!(stem.depth == 0 && start < base_seg_ind) &&
                        !pointInside(turtle.position())) {
                        return false;
                    }
                }
            }

            if (seg_ind > start) {
                // Calc number of splits at this seg (N/A for helix)  
                if (m_params.curve_v[depth] >= 0) {
                    int num_of_splits = 0;
                    if (m_params.base_splits > 0 && depth == 0 && seg_ind == base_seg_ind) {
                        // If base_seg_ind and has base splits then override with base split number  
                        num_of_splits = static_cast<int>(rnd(m_seed) *
                            (m_params.base_splits + 0.5));
                    }
                    else if (seg_splits > 0 && seg_ind < curve_res &&
                        (depth > 0 || seg_ind > base_seg_ind)) {
                        // Otherwise get number of splits from seg_splits and use Floyd-Steinberg  
                        if (rnd(m_seed) <= clone_prob) {
                            num_of_splits = static_cast<int>(seg_splits +
                                m_split_num_error[depth]);
                            m_split_num_error[depth] -= num_of_splits - seg_splits;
                            // Reduce clone/branch propensity  
                            clone_prob /= num_of_splits + 1;
                        }
                    }

                    // Perform cloning if needed  
                    if (num_of_splits > 0) {
                        // Calc angles for split  
                        bool is_base_split = (m_params.base_splits > 0 && depth == 0 &&
                            seg_ind == base_seg_ind);
                        bool using_direct_split = m_params.split_angle[depth] < 0;
                        float spr_angle, spl_angle;

                        if (using_direct_split) {
                            spr_angle = std::abs(m_params.split_angle[depth]) +
                                rnd(m_seed, -1.0f, 1.0f) * m_params.split_angle_v[depth];
                            spl_angle = 0;
                            split_corr_angle = 0;
                        }
                        else {
                            float declination = turtle.direction().declination();
                            spl_angle = m_params.split_angle[depth] +
                                rnd(m_seed, -1.0f, 1.0f) * m_params.split_angle_v[depth] -
                                declination;
                            spl_angle = std::max(0.0f, spl_angle);
                            split_corr_angle = spl_angle / remaining_segs;
                            spr_angle = -(20 + 0.75 * (30 + std::abs(declination - 90) *
                                pow2(rnd(m_seed))));
                        }

                        // Apply split to base stem  
                        turtle.pitchDown(math::radians(spl_angle / 2));  // Convert degrees to radians

                        // Apply spread if splitting to 2 and not base split  
                        if (!is_base_split && num_of_splits == 1) {
                            if (using_direct_split) {
                                turtle.turnLeft(math::radians(spr_angle / 2));  // Convert degrees to radians
                            }
                            else {
                                Quaternion quat(Vec3f(0, 0, 1), math::radians(-spr_angle / 2));
                                turtle.setDirection(normalize(quat.rotate(turtle.direction())));
                                turtle.setRight(normalize(quat.rotate(turtle.right())));
                            }
                        }
                    }
                    else {
                        // Just apply curve and split correction  
                        turtle.turnLeft(math::radians(rnd(m_seed, -1.0f, 1.0f) * m_params.bend_v[depth] / curve_res));  // Convert degrees to radians
                        float curve_angle = calcCurveAngle(depth, seg_ind);  // Already returns radians
                        turtle.pitchDown(curve_angle - split_corr_angle);
                    }

                    // Apply full tropism if not trunk/main branch and horizontal tropism if is  
                    if (depth > 1) {
                        applyTropism(turtle, Vec3f(m_params.tropism[0], m_params.tropism[1], m_params.tropism[2]));
                    }
                    else {
                        applyTropism(turtle, Vec3f(m_params.tropism[0], 0, m_params.tropism[2]));
                    }
                }
            }
        }

        return pointInside(turtle.position());
    }
    
    // ----------------------------------------------------------------------------------------------------------------------------
    void Tree::createBranches()
    {
        std::vector<std::string> level_names = { "Trunk" };
        for (int i = 1; i < m_params.levels; i++) {
            level_names.push_back("Branches " + std::to_string(i));
        }

        for (int level_depth = 0; level_depth < m_params.levels; level_depth++) {
            std::shared_ptr<BezierCurve> level_curve = std::make_shared<BezierCurve>(level_names[level_depth]);
            level_curve->dimensions = CurveDimension::Curve3D;
            level_curve->resolution_u = m_params.curve_res[level_depth];
            level_curve->fill_mode = FillMode::FULL;
            level_curve->bevel_depth = 1.0f;
            level_curve->bevel_resolution = m_params.bevel_res[level_depth];

            m_branch_curves.push_back(level_curve);
        }

        for (int ind = 0; ind < m_params.branches[0]; ind++) {
            m_tree_scale = m_params.g_scale + rnd(m_seed, -1.0f, 1.0f) * m_params.g_scale_v;

            // Initialize turtle
            CHTurtle turtle(Vec3f(0.0f), Vec3f(0.0f, 1.0f, 0.0f), Vec3f(1.0f, 0.0f, 0.0f), 1.0f);
            if (m_params.branches[0] > 1) {
                float angle = math::two_pi * ind / m_params.branches[0];
                turtle.rollRight(angle);
                float radius = m_tree_scale * 0.1f;
                turtle.setPosition(Vec3f(cos(angle) * radius, 0.0f, sin(angle) * radius));
            }
            else {
                turtle.rollRight(rnd(m_seed, 0, math::two_pi));
            }

            // Create trunk spline in level 0 curve
            std::shared_ptr<BezierSpline> trunk_spline = m_branch_curves[0]->addSpline();
            trunk_spline->resolution_u = m_params.curve_res[0];
            trunk_spline->radius_interpolation = RadiusInterpolation::CARDINAL;
            
            // Create trunk stem with the spline
            Stem trunk(0, trunk_spline);
            
            // Add to m_stems FIRST, then build with reference
            // This ensures children are added to the actual stored stem, not a copy
            m_stems.push_back(trunk);
            Stem& trunk_ref = m_stems.back();  // Get reference to stored stem
            
            makeStem(turtle, trunk_ref);  // Build on the stored reference
        }
    }

} // namespace prayground
