#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <algorithm>
#include <string>
#include <cmath>
#include <cstdlib>
using namespace std;


//  Definition for singly-linked list.
struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        std::unordered_map<int, int> h;
        std::vector<int> o_;
        auto N = nums.size();
        for(int i =0; i< N; i++){
            h.insert({nums[i],i});
        }
        for(int i=0;i<N;i++){
            auto ptr = h.find(target-nums[i]);
            if(ptr != h.end() && h[target-nums[i]]!=i){
                o_.push_back(i);
                o_.push_back(h[target-nums[i]]);
                break;
            }
        }
        return o_;
    }
    int romanToInt(string s) {
        int N = s.size();
        auto h = std::unordered_map<int,int>{
            {'I',1},
            {'V',5},
            {'X',10},
            {'L',50},
            {'C',100},
            {'D',500},
            {'M',1000}
        };
        auto mulh = std::unordered_map<std::string, int>{
            {"IV", 4},
            {"IX", 9},
            {"XL", 40},
            {"XC", 90},
            {"CD",400},
            {"CM",900}
        };

        int sum_ = 0;
        int i = N-1;
        while(i>-1){
            if(i==0){
                sum_ += h[s[i]];
                break;
            }
            auto sub_ = s.substr(i-1,2);
            if (mulh.find(sub_) != mulh.end())
            {
                sum_ += mulh[sub_];
                i -= 2;
            }
            else {
                // sum_ += h[sub_[0]];
                sum_ += h[sub_[1]];
                i -=1;
            }
        }            
        return sum_;
    }
    bool hasCycle(ListNode *head) {
        if(head == nullptr){
            return false;
        }
        auto h = std::unordered_map<ListNode*, int>();
        auto left = head/* condition */->next;
        ListNode* cv = head;
        int pos = 0;
        while(left != nullptr){
            if(h.find(cv) != h.end()){
                return true;
            }
            else{
                h.insert({cv, pos});
            }
            cv = left->next;
            left = left->next;
            pos += 1;
        }
        return false;
    }

    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB){
        if (headA==nullptr && headB != nullptr)
        {
            return headB;
        }
        if (headB==nullptr && headA != nullptr)
        {
            return headA;
        }
        if (headA==nullptr && headB==nullptr)
        {
            return nullptr;
        }
        auto ha = std::unordered_set<ListNode*>();
        auto hb = std::unordered_set<ListNode*>();
        auto cpa = headA;
        auto cpb = headB;
        int i = 1;
        while(true){
            if (cpa == nullptr && cpb == nullptr)
            {
                break;
            }
            if(i%2 ==0){
                if (cpa == nullptr && cpb != nullptr)
                {
                    i +=1;
                    continue;
                }
                if (hb.find(cpa) != hb.end())
                {
                    return cpa;
                }
                ha.insert(cpa);
                cpa = cpa->next;
            }
            if(i%2 != 0){
                if (cpb==nullptr && cpa != nullptr)
                {
                    i+=1;
                    continue;
                }
                if (ha.find(cpb) != ha.end())
                {
                    return cpb;
                }
                
                hb.insert(cpb);
                cpb = cpb->next;
            }
            i+=1;
        }
        return nullptr;
    }
    int majorityElement(vector<int>& nums) {
        int N = nums.size();
        auto h = std::unordered_map<int,int>();
        for (size_t i = 0; i < N; i++)
        {
            h.insert({nums[i],0});
        }
        for (size_t i = 0; i < N; i++)
        {
            h[nums[i]] += 1;
        }
        for(auto it: h){
            if(it.second > N/2){
                return it.first;
            }
        }

        
    }

    bool isHappy(int n) {
        // get digits
        // if sum(square(digits)) = 1 then true
        // retry
        int buff[10];
        int powers[10];
        int N =10; 
        for(int i=0;i<N;i++){
            powers[i]=(int)pow((float)10,(float)(N-i-1));
        }
        int tmp = n;
        int maxiter = 1000;
        while(maxiter--){
            for(int i=0;i<N;i++){
                buff[i] = tmp/powers[i];
                tmp = tmp - buff[i]*powers[i];
            }
            for(int i=0;i<N;i++){
                tmp += buff[i]*buff[i];
            }
            if (tmp == 1){
                return true;
            }
        }

        return false;
    }
    bool isIsomorphic(string s, string t) {
        auto hs = std::unordered_map<char, int>();
        auto ht = std::unordered_map<char, int>();
        int Ns = s.size();
        for(int i=0;i<Ns;i++){
            if (hs.find(s[i])==hs.end() && ht.find(t[i]) == ht.end())
            {
                hs[s[i]] = i;
                ht[t[i]] = i;
                continue;

            }
            else if (hs.find(s[i]) == hs.end() && ht.find(t[i]) != ht.end()){
                return false;
            }
            else if (hs.find(s[i]) != hs.end() && ht.find(t[i]) == ht.end()){
                return false;
            }
            else{
                if(hs[s[i]]!=ht[t[i]]){
                    return false;
                }
            }
            hs[s[i]] = i;
            ht[t[i]] = i;
        }

        return true;
    }
    bool containsDuplicate(vector<int>& nums) {
        auto h =  std::unordered_map<int,int>();
        int N = nums.size();
        for(int i=0;i<N;i++){
            h.insert({nums[i],0});
        }
        for(int i=0;i<N;i++){
            h[nums[i]] +=1;
        }
        for(auto el: h){
            if(el.second>1){
                return true;
            }
        }
        return false;
    }

    bool containsNearbyDuplicate(vector<int>& nums, int k) {
        //Given an integer array nums and an integer k, 
        //return true if there are two distinct indices i and j 
        //in the array such that nums[i] == nums[j] and abs(i - j) <= k.
        auto h = std::unordered_map<int, std::vector<int>>();
        int N = nums.size();
        for(int i=0;i<N;i++){
            h.insert({nums[i], std::vector<int>()});
        }
        for(int i=0;i<N;i++){
            h[nums[i]].push_back(i);
        }
        for(auto& el: h){
            for(int i=0;i<el.second.size()-1;i++){
                if(abs(el.second[i]-el.second[i+1]) <= k){
                    return true;
                }
            }
        }
        return false;
    }
    bool isAnagram(string s, string t) {
        
    }

};


int main(){
    auto sol = Solution();
    int a = 123456789;
    std::cout << sol.isHappy(a) << std::endl;
    return 0;
}

