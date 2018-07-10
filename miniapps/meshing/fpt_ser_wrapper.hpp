class findpts_gslib
{
protected: 
//        int dim, nel, qo, msz;
private:
        IntegrationRule ir;
        double *fmesh;
        struct findpts_data_2 *fda;
        struct findpts_data_3 *fdb;
        struct comm cc;
        int dim, nel, qo, msz;

public:
      findpts_gslib (FiniteElementSpace *pfes, Mesh *pmesh, int QORDER);

      void gslib_findpts_setup();

      void gslib_findpts(uint *pcode, uint *pproc, uint *pel,
      double *pr,double *pd,double *xp, double *yp, double *zp, int nxyz);

      void gslib_findpts_eval (double *fieldout, uint *pcode, uint *pproc, uint *pel,
            double *pr,double *fieldin, int nxyz);

      void gslib_findpts_free ();

      ~findpts_gslib();
};

findpts_gslib::findpts_gslib (FiniteElementSpace *pfes, Mesh *pmesh, int QORDER)
{
   const int geom_type = pfes->GetFE(0)->GetGeomType();
   this->ir = IntRulesLo.Get(geom_type, QORDER); 
   dim = pmesh->Dimension();
   nel = pmesh->GetNE();
   qo = sqrt(ir.GetNPoints());
   if (dim==3) qo = cbrt(ir.GetNPoints());
   if (dim==2) 
     {msz = nel*qo*qo;}
   else
     {msz = nel*qo*qo*qo;}
   this->fmesh = new double[dim*msz];

   const int NE = nel, nsp = this->ir.GetNPoints(), NR = qo;
   int np;
   GridFunction nodes(pfes);
   pmesh->GetNodes(nodes);

   if (dim==2)
   {
    np = 0;
    int npt = NE*nsp;
    for (int i = 0; i < NE; i++)
    {
       for (int j = 0; j < nsp; j++)
       {
         const IntegrationPoint &ip = this->ir.IntPoint(j);
         this->fmesh[0+np] = nodes.GetValue(i, ip, 1);
         this->fmesh[npt+np] =nodes.GetValue(i, ip, 2);
         np = np+1;
       }
    }
   } //end dim==2
  else
  {
   np = 0; 
    int npt = NE*nsp;
    for (int i = 0; i < NE; i++)
    {  
       for (int j = 0; j < nsp; j++)
       { 
         const IntegrationPoint &ip = this->ir.IntPoint(j);
         this->fmesh[0+np] = nodes.GetValue(i, ip, 1);
         this->fmesh[npt+np] =nodes.GetValue(i, ip, 2);
         this->fmesh[2*npt+np] =nodes.GetValue(i, ip, 3);
         np = np+1;
       }
    }
   }
}

void findpts_gslib::gslib_findpts_setup()
{
   const int NE = nel, nsp = this->ir.GetNPoints(), NR = qo;
   comm_init(&this->cc,0);
   double bb_t = 0.05;
   int npt_max = 256;
   double tol = 1.e-12;
   int ntot = pow(NR,dim)*NE;
   int npt = NE*NR*NR;
   if (dim==3) {npt *= NR;}

   if (dim==2)
   {
    unsigned nr[2] = {NR,NR};
    unsigned mr[2] = {2*NR,2*NR};
    double *const elx[2] = {&this->fmesh[0],&this->fmesh[npt]};
    this->fda=findpts_setup_2(&this->cc,elx,nr,NE,mr,bb_t,ntot,ntot,npt_max,tol);
   }
   else
   {
    unsigned nr[3] = {NR,NR,NR};
    unsigned mr[3] = {2*NR,2*NR,2*NR};
    double *const elx[3] = {&this->fmesh[0],&this->fmesh[npt],&this->fmesh[2*npt]};
    this->fdb=findpts_setup_3(&this->cc,elx,nr,NE,mr,bb_t,ntot,ntot,npt_max,tol);
   }
}

void findpts_gslib::gslib_findpts(uint *pcode, uint *pproc, uint *pel,double *pr,double *pd,double *xp, double *yp, double *zp, int nxyz)
{
    if (dim==2)
    {
    int npt = nel*qo*qo;
    const double *const elx[2] = {&fmesh[0],&fmesh[npt]};
    const double *xv_base[2];
    xv_base[0]=xp, xv_base[1]=yp;
    unsigned xv_stride[2];
    xv_stride[0] = sizeof(double),
    xv_stride[1] = sizeof(double);
    uint *const code_base = pcode;
    uint *const proc_base = pproc;
    uint *const el_base = pel;
    double *const r_base = pr;
    double *const dist_base = pd;
    findpts_2(
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr,sizeof(double)*dim,
      dist_base,sizeof(double),
      xv_base,     xv_stride,
      nxyz,this->fda);
    }
   else
   {
    int npt = nel*qo*qo*qo;
    const double *const elx[3] = {&fmesh[0],&fmesh[npt],&fmesh[2*npt]};
    const double *xv_base[3];
    xv_base[0]=xp, xv_base[1]=yp;xv_base[2]=zp;
    unsigned xv_stride[3];
    xv_stride[0] = sizeof(double),
    xv_stride[1] = sizeof(double);
    xv_stride[2] = sizeof(double);
    uint *const code_base = pcode;
    uint *const proc_base = pproc;
    uint *const el_base = pel;
    double *const r_base = pr;
    double *const dist_base = pd;
    findpts_3(
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr,sizeof(double)*dim,
      dist_base,sizeof(double),
      xv_base,     xv_stride,
      nxyz,this->fdb);
   }
//   if (this->cc.id==0) {cout <<  "Done findpts\n";}
}

void findpts_gslib::gslib_findpts_eval(
                double *fieldout, uint *pcode, uint *pproc, uint *pel, double *pr,
                   double *fieldin, int nxyz)
{
    if (dim==2)
    {
    int npt = nel*qo*qo;
    const double *const elx[2] = {&fmesh[0],&fmesh[npt]};
    uint *const code_base = pcode;
    uint *const proc_base = pproc;
    uint *const el_base = pel;
    double *const r_base = pr;
    double *const out_base = fieldout;
    double *const in_base = fieldin;
    findpts_eval_2(out_base,sizeof(double),
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr,sizeof(double)*dim,
      nxyz,fieldin,this->fda);
    }
   else
   {
    int npt = nel*qo*qo;
    const double *const elx[3] = {&fmesh[0],&fmesh[npt],&fmesh[2*npt]};
    uint *const code_base = pcode;
    uint *const proc_base = pproc;
    uint *const el_base = pel;
    double *const r_base = pr;
    double *const out_base = fieldout;
    double *const in_base = fieldin;
    findpts_eval_3(out_base,sizeof(double),
      code_base,sizeof(uint),
      proc_base,sizeof(uint),
      el_base,sizeof(uint),
      pr,sizeof(double)*dim,
      nxyz,fieldin,this->fdb);
   }
//   if (this->cc.id==0) {cout <<  "Done findpts_eval\n";}
}

void findpts_gslib::gslib_findpts_free ()
{
 if (dim==2)
 {
  findpts_free_2(this->fda);
 }
 else
 {
  findpts_free_3(this->fdb);
 }
}
