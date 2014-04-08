/*
 ============================================================================
 Name        	:  MPIBilaterImageFilter.c
 Author      	:  Evan Hosseini
 Course		:  CS708 - Scientific Computing
 Description :  MPI Bilateral Image Filter
 ============================================================================
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <memory.h>

// Utility macros
#define max(x, y) ((x>y) ? (x):(y))
#define min(x, y) ((x<y) ? (x):(y))
#define ROOT 0

//#define DEBUG

// Filter parameters
#define N 7			// this parameter should always be an odd number
#define sigma1 10
#define sigma2 40

// Global variables
int xdim;
int ydim;
int maxraw;
unsigned char *image;


void ReadPGM(FILE* fp)
{
	int c;
	int i,j;
	int val;
	unsigned char *line;
	char buf[1024];


	while ((c=fgetc(fp)) == '#')
		fgets(buf, 1024, fp);
	ungetc(c, fp);
	if (fscanf(fp, "P%d\n", &c) != 1) {
		printf ("read error ....");
		exit(0);
	}
	if (c != 5 && c != 2) {
		printf ("read error ....");
		exit(0);
	}

	if (c==5) {
		while ((c=fgetc(fp)) == '#')
			fgets(buf, 1024, fp);
		ungetc(c, fp);
		if (fscanf(fp, "%d%d%d",&xdim, &ydim, &maxraw) != 3) {
			printf("failed to read width/height/max\n");
			exit(0);
		}
		printf("Width=%d, Height=%d \nMaximum=%d\n",xdim,ydim,maxraw);

		image = (unsigned char*)malloc(sizeof(unsigned char)*xdim*ydim);
		getc(fp);

		line = (unsigned char *)malloc(sizeof(unsigned char)*xdim);
		for (j=0; j<ydim; j++) {
			fread(line, 1, xdim, fp);
			for (i=0; i<xdim; i++) {
				image[j*xdim+i] = line[i];
			}
		}
		free(line);

	}

	else if (c==2) {
		while ((c=fgetc(fp)) == '#')
			fgets(buf, 1024, fp);
		ungetc(c, fp);
		if (fscanf(fp, "%d%d%d", &xdim, &ydim, &maxraw) != 3) {
			printf("failed to read width/height/max\n");
			exit(0);
		}
		printf("Width=%d, Height=%d \nMaximum=%d,\n",xdim,ydim,maxraw);

		image = (unsigned char*)malloc(sizeof(unsigned char)*xdim*ydim);
		getc(fp);

		for (j=0; j<ydim; j++)
			for (i=0; i<xdim; i++) {
				fscanf(fp, "%d", &val);
				image[j*xdim+i] = val;
			}

	}
	fclose(fp);
}

void WritePGM(FILE* fp)
{
	int i,j;

	fprintf(fp, "P5\n%d %d\n%d\n", xdim, ydim, 255);

	for (j=0; j<ydim; j++)
		for (i=0; i<xdim; i++)
			fputc(image[j*xdim+i], fp);

	printf("Finished writing to output file! \n");

	fclose(fp);
}

// Bilateral Filtering Function - uses filter parameters defined at beginning of this source
unsigned char cBilateralFilter( unsigned char *pIterator, int x, int y )
{
	float result = 0;
	int m, m_dim, n, n_dim;
	float w_sum = 0;
	int left_bound = N/2;
	int right_bound = N/2;
	int top_bound = N/2;
	int bottom_bound = N/2;

	// Determine if given pixel window overlaps image boundary and scale appropriately
	if ( x - N/2 < 0 )
		left_bound -= (N/2 - x);
	if ( x + N/2 > xdim )
		right_bound -= (x + N/2 - xdim);
	if ( y - N/2 < 0 )
		top_bound -= (N/2 - y);
	if ( y + N/2 > ydim )
		bottom_bound -= (y + N/2 - ydim);

	m_dim = left_bound + right_bound +1;
	n_dim = top_bound + bottom_bound + 1;

	float w_raw[m_dim][n_dim];
	float w_norm[m_dim][n_dim];
	float m_num1 = 0, m_num2 = 0, I_mn = 0, I_ij = 0;

	// Loop over all members of the window and calculate weight factors
	for ( n = 0; n < n_dim; n++ )
		for ( m = 0; m < m_dim; m++)
		{
			// Calculate Gaussian spatial numerator
			m_num1 = sigma2 * ( powf( m - left_bound, 2 ) + powf( n - top_bound, 2 ) );

			// Calculate Gaussian intensity numerator
			I_mn = (float) *(pIterator - left_bound + m + xdim * (n - top_bound));
			I_ij = (float) *pIterator;
			m_num2 = sigma1 * powf( I_mn - I_ij, 2 );

			// Calculate weight value and update cumulative sum for normalization
			w_raw[m][n] = expf( - ( ( m_num1 + m_num2 ) / ( powf( sigma1, 2 ) * powf( sigma2, 2 ) ) ) );
			w_sum += w_raw[m][n];
		}

	// Normalize weight values and apply to the pixel
	for (n = 0; n < n_dim; n++)
		for (m = 0; m < m_dim; m++)
		{
			w_norm[m][n] = w_raw[m][n] / w_sum;
			result += *(pIterator - left_bound + m + xdim * (n - top_bound)) * w_norm[m][n];
		}

//	result /= (m_dim * n_dim);
	return (unsigned char) result;
}

int main(int argc, char *argv[])
{
	int					my_rank = 0;										// rank of process
	int					num_procs = 1;									// number of processes
	int					x, y;														// image and sub-image indices
	int					end_index, start_index = 0;				// sub-division based node index
	FILE					*fp;														// image file pointer
	unsigned char *pSubResult, *pSubRef;						// sub-result pointer and reference
	unsigned char *pImageItr;											// local image iterator
	int					s = 1;													// sub-division parameter
	double     		dStartTime = 0, dEndTime = 0;			// time stamps for bench mark



	/* start up MPI */
	MPI_Init(&argc, &argv);

	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 

	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

	// Master node setup
	if (my_rank == ROOT)
	{
		// Input Checking
		if (argc != 3 && argc != 4)
		{
			printf("Incorrect input format! \n");
			printf("Usage: MyProgram <input_ppm> <output_ppm> <sub-division parameter (optional)> \n");
			printf("       <input_ppm>: PGM file \n");
			printf("       <output_ppm>: PGM file \n");
			MPI_Finalize();
			return 0;
		}

		// Assign sub-division parameter if passed, else by default set to 1
		if (argc == 4)
			s = atoi( argv[3]);
		else
			s = 1;

		// Enforce s parameter validity
		if ( s != 1 && s != 2 && s != 4)
		{
			printf("Only can pass value of 1,2, or 4 to sub-division parameter! \n");
			MPI_Finalize();
			return 0;
		}

		// Enforce mpirun given correct processor number(s)
		if ( s * s != num_procs)
		{
			printf("Must give mpirun command s^2 processors to corresponding sub-division parameter argument passed to the program! \n");
			MPI_Finalize();
			return 0;
		}

		/* begin reading PGM.... */
		printf("begin reading PGM.... \n");
		if ((fp=fopen(argv[1], "r"))==NULL)
		{
			printf("read error...\n");
			MPI_Finalize();
			return 0;
		}
		ReadPGM(fp);

		// Generate start time stamp
		dStartTime = MPI_Wtime();
	}

	// Broadcast the input image to all nodes
	MPI_Bcast( image, xdim * ydim, MPI_CHAR, ROOT, MPI_COMM_WORLD );
	//printf( "Broadcast return code = %d from node %d \n" , MPI_Bcast( image, xdim * ydim, MPI_CHAR, ROOT, MPI_COMM_WORLD ), my_rank );

	MPI_Barrier( MPI_COMM_WORLD );

	printf( "So far so good boss, @ line %d \n", __LINE__ );

	// Calculate each processor's work partition
	start_index = my_rank * (ydim / num_procs);

	// Last node is burdened w/ processing leftover sub-division entries
	if (my_rank == num_procs - 1)
		end_index = ydim;
	else
		end_index = (my_rank + 1) * (ydim / num_procs);

	// Allocate memory for sub-result buffer
	pSubResult = malloc( sizeof(unsigned char) * ( end_index - start_index) * xdim );
	pSubRef = pSubResult;

	// Initialize the iterator to the first image element and offset by start_index
	pImageItr = image;
	pImageItr += start_index * xdim;

	// Loop over assigned workload and dump result to local buffer
	for ( y = start_index; y < end_index; y++ )
		for ( x = 0; x < xdim; x++, pSubResult++, pImageItr++ )
		{
			(*pSubResult) = cBilateralFilter( pImageItr, x, y );
#ifdef DEBUG
			if (my_rank == ROOT)
				printf( "Value of pSubResult @ %x is %d! \n", pSubResult, *pSubResult );
#endif
		}

	// MPI gatherv container for receive message sizes and displacements
	int recvcounts[num_procs];
	int displs[num_procs];
	int k;
	for (k = 0; k < num_procs; k++)
	{
		recvcounts[k] = ydim / num_procs * xdim;
		displs[k] = k * recvcounts[k];
	}
	recvcounts[num_procs - 1] += ydim % num_procs * xdim;

	// Synchronize before allowing root node to collect final image
	MPI_Barrier( MPI_COMM_WORLD );

	printf( "So far so good boss, @ line %d \n", __LINE__ );

	// Collect sub-result data from all nodes
	MPI_Gatherv( pSubRef, (end_index - start_index) * xdim, MPI_CHAR, image,
						   recvcounts, displs, MPI_CHAR, ROOT, MPI_COMM_WORLD );

	printf( "So far so good boss, @ line %d \n", __LINE__ );

	if ( my_rank == ROOT )
	{
		dEndTime = MPI_Wtime();
		printf( "Total MPI Execution time = %3.2f seconds \n", dEndTime - dStartTime );
	}
	// Allow nodes to release sub-result memory
	free( pSubRef );

	if ( my_rank == ROOT )
	{
		// Begin writing PGM....
		printf("Begin writing PGM.... \n");
		if ((fp=fopen(argv[2], "wb")) == NULL){
			printf("write pgm error....\n");
			MPI_Finalize();
			return 0;
		}
		WritePGM(fp);

		free(image);
	}

	/* shut down MPI */
	MPI_Finalize();

	return 0;
}
