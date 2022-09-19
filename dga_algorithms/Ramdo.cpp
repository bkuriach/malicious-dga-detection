#include <iostream>
#include <string>


int main(int argc, const char * argv[]) {
    
    FILE* fp;
    if ((fp = fopen("ramdo.txt", "w+")) == NULL) {
        exit(1);
    }
    int numdoms =0;
    unsigned int initial_seed = 0xDEADBEEF;
    int domain_iterator = 0;
    while(numdoms < 10)
    {
        unsigned int xor1 = 0;
        unsigned int shl = initial_seed << 1;
        domain_iterator += 1;
        unsigned int step1 = domain_iterator * shl;
        unsigned int step1b = domain_iterator * initial_seed;
        domain_iterator -= 1;
        unsigned int iter_seed = domain_iterator * initial_seed;
        unsigned int imul_edx = iter_seed * 0x1a;
        xor1 = step1 ^ imul_edx;
        int domain_length = 0;
        char domain_name[0x20];
        memset(domain_name, 0, 0x20);

        while(domain_length < 0x10)
        {
            unsigned int xor1_divide = xor1 / 0x1a;
            unsigned int xor1_remainder = xor1 % 0x1a;
            unsigned int xo1_rem_20 = xor1_remainder + 0x20;
            unsigned int xo1_step2 = xo1_rem_20 ^ 0xa1;
            unsigned int dom_byte = 0x41 + (0xa1 ^ xo1_step2);
            char dom[100];
            sprintf(dom, "%c", dom_byte);
            printf("%s", dom);
            domain_name[domain_length] = dom_byte;
            unsigned int imul_iter = domain_length * step1;
            unsigned int imul_result = domain_length * imul_iter;
            unsigned int imul_1a = 0x1a * imul_result;
            unsigned int xor2 = xor1 ^ imul_1a;
            xor1 = xor1 + xor2;
            domain_length += 1;
        }
        printf(".org\n");
        strcat(domain_name, ".org");
        fprintf(fp, "%s\n", domain_name);
        memset(domain_name, 0, strlen(domain_name));

        domain_iterator +=1;
        numdoms +=1;
    }
    fclose(fp);

    return 0;
}
