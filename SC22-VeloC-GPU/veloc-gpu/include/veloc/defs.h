#ifndef __DEFS_H
#define __DEFS_H

/*---------------------------------------------------------------------------
                                  Defines
---------------------------------------------------------------------------*/

/** Defines */

#ifndef VELOC_SUCCESS
#define VELOC_SUCCESS (0)
#endif

#ifndef VELOC_FAILURE
#define VELOC_FAILURE (-1)
#endif

#define VELOC_MAX_NAME (1024)

#define VELOC_RECOVER_ALL (0)
#define VELOC_RECOVER_SOME (1)
#define VELOC_RECOVER_REST (2)

#define VELOC_CKPT_ALL (0)
#define VELOC_CKPT_SOME (1)
#define VELOC_CKPT_REST (2)

#define DEFAULT (0)
#define READ_ONLY (1)
typedef int (*release_routine)();

#define NO_PREFETCH (0)
#define PREFETCH_STARTED (1)
#define PREFETCH_COMPLETED (2)
#define PREFETCH_CONSUMED (3)

#define HOST_MEM (0)
#define GPU_MEM (1)

#define INIT_REGION (0)
#define TRF_IN_PROGRESS (1)
#define TRF_COMPLETED (2)

#define MAX_VERSIONS_PER_SHOT (1000000)
#define MAX_REGIONS_PER_CKPT (10)

#endif //__DEFS_H
