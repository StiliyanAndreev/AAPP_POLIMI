#include "data.hpp"
#include "process.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <vector>

extern "C" {
#include "mpi_error_check.h"
}

int main(int argc, char* argv[]) {
  int provided_thread_level;
  const int rc_init = MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided_thread_level);
  exit_on_fail(rc_init);
  
  if (provided_thread_level < MPI_THREAD_SINGLE) {
    std::cerr << "Minimum MPI level not satisfied." << std::endl;
    return EXIT_FAILURE;
  }

  int world_rank = 0;
  int world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  MPI_Datatype mpi_molecule_type = chem::create_molecule_MPI_type();

  //==---------------------------------------------------------------------------------------------------==//
  // DATA GENERATION PHASE
  //==---------------------------------------------------------------------------------------------------==//
  
  auto data = std::vector<chem::molecule>{};
  std::uint32_t num_data = 0;

  if (world_rank == 0) {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " num_data" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    auto parser = std::istringstream(argv[1]);
    parser >> num_data;
    if (parser.fail()) {
      std::cerr << "Error: unable to understand the number \"" << argv[1] << '"' << std::endl;
      MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    data = chem::generate_data(num_data);
  }

  // Broadcast total number of molecules
  MPI_Bcast(&num_data, 1, MPI_UINT32_T, 0, MPI_COMM_WORLD);

  //==---------------------------------------------------------------------------------------------------==//
  // DISTRIBUTION PHASE
  //==---------------------------------------------------------------------------------------------------==//

  // Challenge assumption: num_data is a multiple of world_size
  const std::size_t local_count = num_data / world_size;
  std::vector<chem::molecule> local_data(local_count);

  MPI_Scatter(data.data(), local_count, mpi_molecule_type, 
              local_data.data(), local_count, mpi_molecule_type, 
              0, MPI_COMM_WORLD);

  //==---------------------------------------------------------------------------------------------------==//
  // COMPUTATION PHASE
  //==---------------------------------------------------------------------------------------------------==//

  for (auto& molecule: local_data) { 
      chem::score(molecule); 
  }

  //==---------------------------------------------------------------------------------------------------==//
  // PARALLEL REDUCTION PHASE (MANUAL TREE REDUCTION)
  //==---------------------------------------------------------------------------------------------------==//

  // Calculate the target size (Top 1%)
  const std::size_t global_top_k_count = 
      std::max(static_cast<std::size_t>(static_cast<float>(num_data) * 0.01f), std::size_t{1});

  // Step 1: Reduce locally first to minimize network traffic
  if (local_data.size() > global_top_k_count) {
      std::partial_sort(local_data.begin(), 
                        local_data.begin() + global_top_k_count, 
                        local_data.end());
      local_data.resize(global_top_k_count);
  } else {
      std::sort(local_data.begin(), local_data.end());
  }

  // Step 2: Binary Tree Reduction
  // At each step, half the active processes send their best data to the other half.
  for (int step = 1; step < world_size; step *= 2) {
      if (world_rank % (2 * step) == 0) {
          int source_rank = world_rank + step;
          
          if (source_rank < world_size) {
              // Prepare buffer for incoming data
              std::vector<chem::molecule> incoming_data(global_top_k_count);
              MPI_Status status;
              
              // Receive data from the neighbor in the tree
              MPI_Recv(incoming_data.data(), global_top_k_count, mpi_molecule_type, 
                       source_rank, 0, MPI_COMM_WORLD, &status);

              // Get actual count received (could be less than top_k if input is small)
              int received_count = 0;
              MPI_Get_count(&status, mpi_molecule_type, &received_count);
              incoming_data.resize(received_count);

              // Merge incoming data with local data
              local_data.insert(local_data.end(), incoming_data.begin(), incoming_data.end());
              
              // Sort and prune again to keep only the best K
              if (local_data.size() > global_top_k_count) {
                   std::partial_sort(local_data.begin(), 
                                     local_data.begin() + global_top_k_count, 
                                     local_data.end());
                   local_data.resize(global_top_k_count);
              } else {
                   std::sort(local_data.begin(), local_data.end());
              }
          }
      } else {
          // Send local top candidates to parent in tree and exit
          int dest_rank = world_rank - step;
          MPI_Send(local_data.data(), local_data.size(), mpi_molecule_type, 
                   dest_rank, 0, MPI_COMM_WORLD);
          break;
      }
  }

  //==---------------------------------------------------------------------------------------------------==//
  // OUTPUT PHASE
  //==---------------------------------------------------------------------------------------------------==//

  // Only Rank 0 has the final reduced result
  if (world_rank == 0) {
      print_data(local_data);
  }

  MPI_Type_free(&mpi_molecule_type);

  const int rc_finalize = MPI_Finalize();
  exit_on_fail(rc_finalize);

  return EXIT_SUCCESS;
}