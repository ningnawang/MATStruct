#include "medial_struct.h"

#include <assert.h>

#include <queue>
#include <stack>

// must call RPD3D_Wrapper::cluster_medges_type() ahead
// to make sure MedialEdge::is_intf is updated
void trace_junction(MedialMesh& mmesh, bool is_debug) {
  assert(mmesh.vertices != nullptr);
  if (is_debug) printf("calling trace_junction...\n");
  for (auto& msphere : *mmesh.vertices) {
    if (msphere.is_deleted) continue;
    int num_medge_intf_adj = 0;
    for (int eid : msphere.edges_) {
      if (mmesh.edges.at(eid).is_intf) num_medge_intf_adj++;
    }
    if (num_medge_intf_adj < 3) continue;
    // if (is_debug)
    //   printf(
    //       "[mark_one_junction_type] marking junction %d, has adj edges %zu, "
    //       "num_medge_intf_adj: %d\n",
    //       msphere.id, msphere.edges_.size(), num_medge_intf_adj);
    MedialStruct mstruct(mmesh.mstructure.size(), MedialType::JUNCTION);
    mstruct.m_sphere_ids.insert(msphere.id);
    // msphere.mstruct_ids.insert(mstruct.id);
    msphere.mstruct_jun_id = mstruct.id;
    mmesh.mstructure.push_back(mstruct);
    if (is_debug)
      printf(
          "[mark_one_junction_type] sphere %d is a real junction, "
          "num_medge_intf_adj: %d, save mstruct: %d\n",
          msphere.id, num_medge_intf_adj, mstruct.id);
  }

  if (is_debug)
    printf("[trace_junction] saved total junctions %zu\n",
           mmesh.mstructure.size());
}

// trace internal/external feature
void trace_singular(MedialMesh& mmesh, bool is_debug) {
  std::set<int> visited;
  if (is_debug) printf("calling trace_singular...\n");
  for (int eid = 0; eid < mmesh.edges.size(); eid++) {
    auto& medge = mmesh.edges.at(eid);
    if (!medge.is_intf && !medge.is_extf) continue;
    if (medge.is_deleted) continue;
    if (visited.find(eid) != visited.end()) continue;
    visited.insert(eid);
    if (is_debug) {
      printf("[trace_singular]===start processing feature medge %d (%d,%d)\n",
             eid, medge.vertices_[0], medge.vertices_[1]);
    }

    // start tracing one feature medge
    // create a new MedialStruct
    MedialStruct mstruct(mmesh.mstructure.size(), MedialType::SEAM);
    std::queue<int> queue;
    queue.push(eid);
    if (is_debug) printf("[trace_singular]---tracing mstruct %d\n", mstruct.id);
    while (!queue.empty()) {
      int cur_eid = queue.front();
      queue.pop();
      if (cur_eid != eid && visited.find(cur_eid) != visited.end()) continue;
      visited.insert(cur_eid);
      auto& cur_medge = mmesh.edges.at(cur_eid);
      if (is_debug)
        printf("[trace_singular] processing eid %d (%d,%d)\n", cur_eid,
               cur_medge.vertices_[0], cur_medge.vertices_[1]);
      assert(cur_medge.is_intf || cur_medge.is_extf);
      if (cur_medge.is_intf) mstruct.type = MedialType::SEAM;
      if (cur_medge.is_extf) mstruct.type = MedialType::BOUNDARY;
      mstruct.m_edge_ids.insert(cur_eid);
      cur_medge.mstruct_id = mstruct.id;
      if (is_debug)
        printf("[trace_singular] medge %d is on intf/extf\n", cur_eid);

      // push neighbors
      for (int i = 0; i < 2; i++) {
        int vid = cur_medge.vertices_[i];
        const auto& cur_msphere = mmesh.vertices->at(vid);
        // reach the junction, NOT add to queue
        if (cur_msphere.is_on_mstruct_junction()) continue;
        for (int adj_eid : cur_msphere.edges_) {
          const auto& adj_medge = mmesh.edges.at(adj_eid);
          if (adj_medge.is_deleted) continue;
          if (visited.find(adj_eid) != visited.end()) continue;
          if ((cur_medge.is_intf && adj_medge.is_intf) ||
              (cur_medge.is_extf && adj_medge.is_extf)) {
            queue.push(adj_eid);
            if (is_debug)
              printf("[trace_singular] pushing adj_medge %d (%d,%d) to queue\n",
                     adj_eid, adj_medge.vertices_[0], adj_medge.vertices_[1]);
          }
        }
      }
    }  // while queue

    // push to medial structure
    mmesh.mstructure.push_back(mstruct);
    if (is_debug) {
      printf("[trace_singular] saved mstruct %d, type %d\n", mstruct.id,
             mstruct.type);
      mmesh.mstructure.back().print_info();
    }
  }  // for
}

void trace_sheet(MedialMesh& mmesh, bool is_debug) {
  if (is_debug) printf("calling trace_sheet...\n");
  std::set<int> visited;
  for (int fid = 0; fid < mmesh.faces.size(); fid++) {
    auto& mface = mmesh.faces.at(fid);
    if (mface.is_deleted) continue;
    if (visited.find(fid) != visited.end()) continue;
    visited.insert(fid);
    if (is_debug) {
      printf("[trace_sheet] start tracing sheet mface %d\n", fid);
    }

    // start tracing one sheet
    // create a new MedialStruct
    MedialStruct mstruct(mmesh.mstructure.size(), MedialType::SHEET);
    std::queue<int> queue;
    queue.push(fid);
    if (is_debug) printf("[trace_sheet] pushing mstruct %d\n", mstruct.id);
    while (!queue.empty()) {
      int cur_fid = queue.front();
      queue.pop();
      if (cur_fid != fid && visited.find(cur_fid) != visited.end()) continue;
      visited.insert(cur_fid);
      auto& cur_mface = mmesh.faces.at(cur_fid);
      if (is_debug) printf("[trace_sheet] queue processing fid %d\n", cur_fid);
      assert(!cur_mface.is_deleted);
      if (!cur_mface.is_on_same_sheet) continue;

      // add mface to mstruct
      mstruct.m_face_ids.insert(cur_fid);
      cur_mface.mstruct_id = mstruct.id;
      if (is_debug)
        printf("[trace_sheet] mface %d is on sheet, add to mstruct %d\n",
               cur_fid, mstruct.id);

      // push neighbor adj_mfaces
      for (int eid : cur_mface.edges_) {
        const auto& medge = mmesh.edges.at(eid);
        if (medge.is_deleted) continue;
        if (medge.is_extf || medge.is_intf) continue;
        if (!medge.is_on_same_sheet) continue;
        for (int adj_fid : medge.faces_) {
          if (adj_fid == cur_fid) continue;
          if (visited.find(adj_fid) != visited.end()) continue;
          aint2 f1f2 = {cur_fid, adj_fid};
          std::sort(f1f2.begin(), f1f2.end());
          int f1f2_idx =
              get_upper_tri_matrix_idx(f1f2[0], f1f2[1], mmesh.faces.size());
          // mface and adj_mface must be on the same sheet to expand
          // assert(mmesh.is_two_faces_on_the_same_sheet[f1f2_idx] != -1);
          if (mmesh.is_two_faces_on_the_same_sheet[f1f2_idx] != 1) continue;
          const auto& adj_mface = mmesh.faces.at(adj_fid);
          if (adj_mface.is_deleted) continue;
          queue.push(adj_fid);
          if (is_debug)
            printf(
                "[trace_sheet] mface %d has adj_mface %d on the same sheet, "
                "add to queue for mstruct %d\n",
                cur_fid, adj_fid, mstruct.id);
        }  // for adj_fid
      }  // for eid
    }  // while queue

    // push to medial structure
    mmesh.mstructure.push_back(mstruct);
  }  // for
}

// trace internal/external feature
void trace_singular_from_endpoints(MedialMesh& mmesh, bool is_debug) {
  std::set<int> evisited;  // visted edges
  if (is_debug) printf("calling trace_singular...\n");

  // Return:
  // TRACE_CODE::TRACE_UNKNOWN: not a valid feature
  // TRACE_CODE::TRACE_TER_ENDPOINT: a valid feature and terminate at endpoint
  // TRACE_CODE::TRACE_TER_LOOP: a valid feature and terminate at loop
  //
  // DFS stack for tracing ONE whole feature from one endpoint
  // format <eid, start_vid>
  std::stack<aint2> stack;
  auto is_trace_one_curve = [&](const aint2& start_ev, MedialStruct& mstruct) {
    while (!stack.empty()) stack.pop();  // clear stack
    stack.push(start_ev);
    // start tracing ONE feature medge
    while (!stack.empty()) {
      int cur_eid = stack.top()[0];
      int cur_vid = stack.top()[1];
      stack.pop();
      if (evisited.find(cur_eid) != evisited.end()) continue;
      evisited.insert(cur_eid);
      const auto& cur_medge = mmesh.edges.at(cur_eid);
      int next_vid = cur_medge.vertices_[0] == cur_vid ? cur_medge.vertices_[1]
                                                       : cur_medge.vertices_[0];
      assert(cur_medge.is_intf || cur_medge.is_extf);
      // update mstruct
      if (cur_medge.is_intf) mstruct.type = MedialType::SEAM;
      if (cur_medge.is_extf) mstruct.type = MedialType::BOUNDARY;
      mstruct.m_edge_ids.insert(cur_eid);
      if (is_debug)
        printf("[trace_singular] mstruct %d add eid %d, next_vid %d\n",
               mstruct.id, cur_eid, next_vid);

      // check if terminate at another endpoint
      const auto& next_mvertex = mmesh.vertices->at(next_vid);
      if (next_mvertex.is_on_seam_endpoint()) {
        if (is_debug)
          printf("[trace_singular] mstruct %d terminate at endpoint %d\n",
                 mstruct.id, next_vid);
        return TRACE_CODE::TRACE_TER_ENDPOINT;  // terminate at endpoint
      }
      if (next_mvertex.id == start_ev[1]) {
        if (is_debug)
          printf(
              "[trace_singular] mstruct %d terminate at loop start_ev "
              "(%d,%d)\n",
              next_mvertex.id, start_ev[0], start_ev[1]);
        return TRACE_CODE::TRACE_TER_LOOP;  // terminate at loop
      }

      // choose next walking direction from all candidate neighbors
      // step 1: store all candiate neighbors
      std::vector<aint2> next_evs;
      for (int next_eid : next_mvertex.edges_) {
        if (next_eid == cur_eid) continue;
        const auto& next_medge = mmesh.edges.at(next_eid);
        if (next_medge.is_deleted) continue;
        if (evisited.find(next_eid) != evisited.end()) continue;
        // TODO: if multiple neighbors to push, which means multiple feature
        // curves to trace, then we need to choose only one.
        // The choice can be made based on the walking direction
        if ((cur_medge.is_intf && next_medge.is_intf) ||
            (cur_medge.is_extf && next_medge.is_extf)) {
          // stack.push({{next_eid, next_vid}});
          next_evs.push_back({{next_eid, next_vid}});
        }
      }  // for next_mvertex.edges_
      if (next_evs.empty()) {
        // continue;  // no neighbors to walk
        if (is_debug)
          printf("[trace_singular] mstruct %d terminate at unknown\n",
                 mstruct.id);
        return TRACE_CODE::TRACE_UNKNOWN;
      }
      if (next_evs.size() == 1) {
        stack.push(next_evs[0]);
        continue;
      }
      // step 2: choose one with the closest walking direction
      // get current curve direction
      const auto& cur_msphere = mmesh.vertices->at(cur_vid);
      const auto& next_msphere = mmesh.vertices->at(next_vid);
      const Vector3 curve_dir =
          GEO::normalize(next_msphere.center - cur_msphere.center);
      // get the next walking direction
      std::vector<double> dot_products;
      for (const auto& next_ev : next_evs) {
        const auto& next_medge = mmesh.edges.at(next_ev[0]);
        int next_next_vid = next_medge.vertices_[0] == next_ev[1]
                                ? next_medge.vertices_[1]
                                : next_medge.vertices_[0];
        const auto& next_next_msphere = mmesh.vertices->at(next_next_vid);
        const Vector3 next_curve_dir =
            GEO::normalize(next_next_msphere.center - next_msphere.center);
        dot_products.push_back(GEO::dot(curve_dir, next_curve_dir));
        if (is_debug)
          printf("[trace_singular] next_next_vid %d, dot %f\n", next_next_vid,
                 dot_products.back());
      }
      // find the max dot product (smallest angle)
      int max_idx = std::distance(
          dot_products.begin(),
          std::max_element(dot_products.begin(), dot_products.end()));
      stack.push(next_evs[max_idx]);  // push the closest walking direction
      printf("[trace_singular] choose max_idx %d with (%d, %d)\n", max_idx,
             next_evs[max_idx][0], next_evs[max_idx][1]);
    }  // while stack

    // if (is_debug)
    //   printf("[trace_singular] mstruct %d terminate at unknown\n",
    //   mstruct.id);
    return TRACE_CODE::TRACE_UNKNOWN;
  };
  //-------------------------------------------------------------------------
  // RULES:
  // 1. endpoints, either junction or convex corner
  // 1.1. two open INTF curves can only be intersected at 1 endpoint
  // 1.2. one open INTF curve has only 2 endpoints
  // 2. the closed INTF curve have 0 endpoints
  /////
  // rule 1: starts/ends at endpoints
  for (int vid = 0; vid < mmesh.vertices->size(); vid++) {
    const auto& mvertex = mmesh.vertices->at(vid);
    if (mvertex.is_deleted) continue;
    if (mvertex.is_on_corner())
      printf("[trace_singular] start tracing corner from endpoint %d\n", vid);
    if (!mvertex.is_on_seam_endpoint()) continue;
    if (is_debug)
      printf("[trace_singular] start tracing feature from endpoint %d\n", vid);

    // format <eid, start_vid>
    // queue for storing ALL possible feature starts from this endpoint
    std::stack<aint2> queue;
    // push edge neighbors if on feature
    for (int eid : mvertex.edges_) {
      const auto& medge = mmesh.edges.at(eid);
      if (medge.is_deleted) continue;
      if (evisited.find(eid) != evisited.end()) continue;
      if (mvertex.is_on_corner() && medge.is_extf) {
        queue.push({{eid, vid}});
        printf("[trace_singular] corner endpoint %d, push extf eid %d\n", vid,
               eid);
      }
      if (mvertex.is_on_junction() && medge.is_intf) queue.push({{eid, vid}});
    }

    while (!queue.empty()) {
      const aint2& start_ev = queue.top();
      queue.pop();
      // create a new MedialStruct
      MedialStruct mstruct(mmesh.mstructure.size(), MedialType::SEAM);
      if (is_debug)
        printf("[trace_singular] creating mstruct %d from start_ev (%d,%d)\n",
               mstruct.id, start_ev[0], start_ev[1]);

      TRACE_CODE trace_code = is_trace_one_curve(start_ev, mstruct);
      // save mstruct if meet feature rule
      // if on boundary/extf, then loose the rule 1
      if (trace_code == TRACE_CODE::TRACE_TER_ENDPOINT ||
          mstruct.type == MedialType::BOUNDARY) {
        if (is_debug)
          printf("[trace_singular] save endpoint mstruct %d\n", mstruct.id);
        mmesh.mstructure.push_back(mstruct);
        // update all medges in mstruct
        for (int eid : mstruct.m_edge_ids)
          mmesh.edges.at(eid).mstruct_id = mstruct.id;
      } else {
        // discard mstruct, not meet feature rule
        if (is_debug)
          printf("[trace_singular] discard endpoint mstruct %d\n", mstruct.id);
      }
    }  // while queue for storing ALL possible feature starts from this endpoint
  }  // for mmesh.vertices

  /////
  // rule 2: the closed INTF curve have 0 endpoints
  for (int eid = 0; eid < mmesh.edges.size(); eid++) {
    const auto& medge = mmesh.edges.at(eid);
    if (medge.is_deleted) continue;
    if (evisited.find(eid) != evisited.end()) continue;
    if (!medge.is_intf && !medge.is_extf) continue;
    // create a new MedialStruct
    aint2 start_ev = {{eid, medge.vertices_[0]}};
    MedialStruct mstruct(mmesh.mstructure.size(), MedialType::SEAM);
    TRACE_CODE trace_code = is_trace_one_curve(start_ev, mstruct);
    // save mstruct if meet feature rule for closed curve (case 2)
    if (trace_code == TRACE_CODE::TRACE_TER_LOOP) {
      if (is_debug)
        printf("[trace_singular] save loop mstruct %d\n", mstruct.id);
      mmesh.mstructure.push_back(mstruct);
      // update all medges in mstruct
      for (int eid : mstruct.m_edge_ids)
        mmesh.edges.at(eid).mstruct_id = mstruct.id;
    }
  }  // for mmesh.edges
}

void mark_intf_extf_nonmanifold(MedialMesh& mmesh, bool is_debug) {
  std::set<int> visited;
  if (is_debug) printf("calling mark_intf_extf_nonmanifold...\n");
  for (int eid = 0; eid < mmesh.edges.size(); eid++) {
    // find the nonmanifold edges and expand
    auto& medge = mmesh.edges.at(eid);
    if (visited.find(eid) != visited.end()) continue;
    visited.insert(eid);
    // if (medge.faces_.size() == 1) medge.is_extf = true;
    if (medge.faces_.size() > 2) medge.is_intf = true;
  }
}

void trace_sheet_manifold(MedialMesh& mmesh, bool is_debug) {
  if (is_debug) printf("calling trace_sheet...\n");
  std::set<int> visited;
  for (int fid = 0; fid < mmesh.faces.size(); fid++) {
    auto& mface = mmesh.faces.at(fid);
    if (visited.find(fid) != visited.end()) continue;
    visited.insert(fid);
    if (is_debug) {
      printf("[trace_sheet] start tracing sheet mface %d\n", fid);
    }

    // start tracing one sheet
    // create a new MedialStruct
    MedialStruct mstruct(mmesh.mstructure.size(), MedialType::SHEET);
    std::queue<int> queue;
    queue.push(fid);
    if (is_debug) printf("[trace_sheet] pushing mstruct %d\n", mstruct.id);
    while (!queue.empty()) {
      int cur_fid = queue.front();
      queue.pop();
      if (cur_fid != fid && visited.find(cur_fid) != visited.end()) continue;
      visited.insert(cur_fid);
      auto& cur_mface = mmesh.faces.at(cur_fid);
      if (is_debug) printf("[trace_sheet] processing fid %d\n", cur_fid);
      assert(!cur_mface.is_deleted);

      // add mface to mstruct
      mstruct.m_face_ids.insert(cur_fid);
      cur_mface.mstruct_id = mstruct.id;
      if (is_debug)
        printf("[trace_sheet] mface %d is on sheet, add to mstruct %d\n",
               cur_fid, mstruct.id);

      // push neighbors
      for (int eid : cur_mface.edges_) {
        // only expand to non-deleted, non-extf, non-intf edges
        const auto& medge = mmesh.edges.at(eid);
        if (medge.is_deleted) continue;
        if (medge.faces_.size() > 2) continue;  // nonmanifold, not expand
        for (int adj_fid : medge.faces_) {
          if (adj_fid == cur_fid) continue;
          const auto& adj_mface = mmesh.faces.at(adj_fid);
          if (adj_mface.is_deleted) continue;
          if (visited.find(adj_fid) != visited.end()) continue;
          queue.push(adj_fid);
        }  // for adj_fid
      }  // for eid
    }  // while queue

    // push to medial structure
    mmesh.mstructure.push_back(mstruct);
  }  // for
}

//////////////////////////////////////////////////////////////////////////
// public functions
//////////////////////////////////////////////////////////////////////////
void trace_medial_structure(const TetMesh& tet_mesh, const SurfaceMesh& sf_mesh,
                            RPD3D_Wrapper& rpd3d, MedialMesh& mmesh,
                            bool is_debug) {
  // if (is_debug)
  printf("start trace_medial_structure ....\n");
  mmesh.mstructure.clear();
  trace_junction(mmesh, false /*is_debug*/);
  trace_singular(mmesh, false /*is_debug*/);
  trace_sheet(mmesh, false /*is_debug*/);
}

void trace_medial_structure_nonmanifold(MedialMesh& mmesh, bool is_debug) {
  printf("start trace_medial_structure_nonmanifold ....\n");
  mmesh.mstructure.clear();
  mark_intf_extf_nonmanifold(mmesh, is_debug);
  trace_junction(mmesh, is_debug);
  trace_singular(mmesh, is_debug);
  trace_sheet_manifold(mmesh, is_debug);
}