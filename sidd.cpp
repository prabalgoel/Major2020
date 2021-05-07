#include <bits/stdc++.h>
using namespace std;


int check(string a,string b,int i){
    int k=0;
    for(int j=i;j<a.size();j++,k++){
        if(a[j]==b[k]){
            continue;
        }
        else break;
    }
    return k;
}
int main(){
    string a,b;
    cin>>a>>b;
    vector<int> ans;
    int n=b.size();
    for(int i=0;i<a.size();){
        if(i+n>a.size())break;
        if(check(a,b,i)==b.size()){
            ans.push_back(i);
            i+=n;
        }
        else i++;
    }
    reverse(ans.begin(),ans.end());
    for(auto j:ans) cout<<j<<" ";

}
